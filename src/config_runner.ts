export { ShaderConfig, ConfigRunner };

// The spatial coefficient LUT.
let spatial_coeff_lut: Float32Array = null;

/// A ShaderConfig object represents a set of implementation decisions.
interface ShaderConfig {
  wgsize_x: number;
  wgsize_y: number;
  tilesize_x: number;
  tilesize_y: number;
  const_sigma_domain: boolean;
  const_sigma_range: boolean;
  const_radius: boolean;
  const_width: boolean;
  const_height: boolean;
  input_type: string;
  prefetch: string;
  spatial_coeffs: string;
}

/// A Result object represents the result of running a config.
interface Result {
  fps: number;
  validated: boolean;
}

/// A class used to benchmark and test different shader configurations.
class ConfigRunner {
  // The WebGPU device.
  device: GPUDevice;

  // The filter parameters.
  // TODO: Make the sigma_domain and sigma_range parameters configurable.
  sigma_domain: number = 3.0;
  sigma_range: number = 0.2;
  radius: number = 2;

  // Cycle through different input images to reduce the likelihood of caching.
  static readonly kNumInputImages = 3;

  // The image/texture objects.
  input_image_staging: GPUTexture;
  input_textures: Array<GPUTexture> = new Array<GPUTexture>(ConfigRunner.kNumInputImages);
  output_texture: GPUTexture;
  width: number;
  height: number;

  // The reference result and diff data.
  reference_data: Uint8Array = null;
  diff_data: Uint8Array = null;

  // Callbacks to receive status and runtime updates.
  UpdateStatusCallback: (str: string, color: string) => void;
  UpdateRuntimeCallback: (str: string) => void;

  /// Constructor.
  constructor(device: GPUDevice) {
    this.device = device;

    // Use console.log for status/runtime updates by default.
    this.UpdateStatusCallback = (str, col) => {
      console.log("%c" + str, "color: " + col);
    };
    this.UpdateRuntimeCallback = console.log;
  }

  /// Update the current status.
  UpdateStatus(str: string, color = "#000000") {
    this.UpdateStatusCallback(str, color);
  }

  /// Update the runtime.
  UpdateRuntime(str: string) {
    this.UpdateRuntimeCallback(str);
  }

  /// Set up the input and output textures.
  async SetupTextures(image: ImageBitmap) {
    this.width = image.width;
    this.height = image.height;

    // Copy the input image to a staging texture.
    this.input_image_staging = this.device.createTexture({
      format: "rgba8unorm",
      size: { width: this.width, height: this.height },
      usage:
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING,
    });
    this.device.queue.copyExternalImageToTexture(
      { source: image },
      { texture: this.input_image_staging },
      { width: this.width, height: this.height }
    );

    // Create the input textures and copy the input image to them.
    const commands = this.device.createCommandEncoder();
    for (let i = 0; i < ConfigRunner.kNumInputImages; i++) {
      this.input_textures[i] = this.device.createTexture({
        size: { width: this.width, height: this.height },
        format: "rgba8unorm",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
      });

      // Copy the input image to the input texture.
      commands.copyTextureToTexture(
        { texture: this.input_image_staging },
        { texture: this.input_textures[i] },
        { width: this.width, height: this.height }
      );
    }
    this.device.queue.submit([commands.finish()]);

    // Create the output texture.
    this.output_texture = this.device.createTexture({
      size: { width: this.width, height: this.height },
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING,
    });

    this.reference_data = null;
  }

  /// Change the filter radius.
  SetRadius(radius: number) {
    this.radius = radius;
    this.reference_data = null;
  }

  /// Run the filter for a specific shader config.
  async RunConfig(params: {
    config: ShaderConfig;
    iterations: number;
    test?: boolean;
  }): Promise<Result> {
    this.UpdateStatus("Setting up...");

    const config = params.config;

    // Set up the filter parameters.
    const parameters = this.device.createBuffer({
      size: 20,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    });
    const param_values = parameters.getMappedRange();
    const param_values_f32 = new Float32Array(param_values);
    const param_values_u32 = new Uint32Array(param_values);

    // Set values for any parameters that are not being embedded as constants.
    let uniform_member_index = 0;
    if (!config.const_sigma_domain && config.spatial_coeffs === "inline") {
      param_values_f32[uniform_member_index++] = -0.5 / (this.sigma_domain * this.sigma_domain);
    }
    if (!config.const_sigma_range) {
      param_values_f32[uniform_member_index++] = -0.5 / (this.sigma_range * this.sigma_range);
    }
    if (!config.const_radius) {
      param_values_u32[uniform_member_index++] = this.radius;
    }
    if (!config.const_width) {
      param_values_u32[uniform_member_index++] = this.width;
    }
    if (!config.const_height) {
      param_values_u32[uniform_member_index++] = this.height;
    }
    parameters.unmap();

    // Generate the shader and create the compute pipeline.
    const module = this.device.createShaderModule({ code: this.GenerateShader(config) });
    const pipeline = this.device.createComputePipeline({
      compute: { module, entryPoint: "main" },
      layout: "auto",
    });

    // Create a bind group for group index 0 for each input image.
    const bind_group_0 = [];
    for (let i = 0; i < ConfigRunner.kNumInputImages; i++) {
      let entries: GPUBindGroupEntry[];
      entries = [
        { binding: 0, resource: this.input_textures[i].createView() },
        { binding: 1, resource: this.output_texture.createView() },
      ];
      if (config.input_type === "image_sample") {
        entries.push({
          binding: 2,
          resource: this.device.createSampler({
            addressModeU: "clamp-to-edge",
            addressModeV: "clamp-to-edge",
            minFilter: "nearest",
            magFilter: "nearest",
          }),
        });
      }
      bind_group_0.push(
        this.device.createBindGroup({
          entries,
          layout: pipeline.getBindGroupLayout(0),
        })
      );
    }

    // Create a uniform buffer for the spatial coefficient LUT if necessary.
    let spatial_coeff_lut_buffer = null;
    if (config.spatial_coeffs === "lut_uniform") {
      let buffer_size = spatial_coeff_lut.length * 16;
      spatial_coeff_lut_buffer = this.device.createBuffer({
        size: buffer_size,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true,
      });

      const dst = new Float32Array(spatial_coeff_lut_buffer.getMappedRange());
      for (let i = 0; i < spatial_coeff_lut.length; i++) {
        dst[i * 4] = spatial_coeff_lut[i];
      }
      spatial_coeff_lut_buffer.unmap();
    }

    // Create the bind group for the uniform parameters if necessary.
    let bind_group_1 = null;
    let bind_group_1_entries = [];
    if (uniform_member_index > 0) {
      bind_group_1_entries.push({ binding: 0, resource: { buffer: parameters } });
    }
    if (spatial_coeff_lut_buffer) {
      bind_group_1_entries.push({ binding: 1, resource: { buffer: spatial_coeff_lut_buffer } });
    }
    if (bind_group_1_entries.length > 0) {
      bind_group_1 = this.device.createBindGroup({
        entries: bind_group_1_entries,
        layout: pipeline.getBindGroupLayout(1),
      });
    }

    // Determine the number of workgroups.
    const pixels_per_group_x = config.wgsize_x * config.tilesize_x;
    const pixels_per_group_y = config.wgsize_y * config.tilesize_y;
    const group_count_x = Math.floor((this.width + pixels_per_group_x - 1) / pixels_per_group_x);
    const group_count_y = Math.floor((this.height + pixels_per_group_y - 1) / pixels_per_group_y);

    // Helper to enqueue `n` back-to-back runs of the shader.
    let Enqueue = (n: number) => {
      const commands = this.device.createCommandEncoder();
      const pass = commands.beginComputePass();
      pass.setPipeline(pipeline);
      if (bind_group_1) {
        // Only set the bind group for the uniform parameters if it was created.
        pass.setBindGroup(1, bind_group_1);
      }
      for (let i = 0; i < n; i++) {
        pass.setBindGroup(0, bind_group_0[n % ConfigRunner.kNumInputImages]);
        pass.dispatchWorkgroups(group_count_x, group_count_y);
      }
      pass.end();
      this.device.queue.submit([commands.finish()]);
    };

    // Warm up run.
    Enqueue(1);
    await this.device.queue.onSubmittedWorkDone();

    let fps = 0;
    if (!params.test) {
      // Timed runs.
      this.UpdateStatus("Running...");
      const start = performance.now();
      Enqueue(params.iterations);
      await this.device.queue.onSubmittedWorkDone();
      const end = performance.now();
      const elapsed = end - start;
      fps = (params.iterations / elapsed) * 1000;
      this.UpdateRuntime(
        `Elapsed time: ${elapsed.toFixed(2)} ms (${fps.toFixed(2)} frames/second)`
      );
    }

    return {
      fps,
      validated: await this.VerifyResult({
        output: this.output_texture,
        config,
        quick: params.test,
      }),
    };
  }

  /// Generate the WGSL shader.
  GenerateShader(config: ShaderConfig): string {
    let indent = 0;
    let wgsl = "";
    let constants = "";
    let structures = "";
    let uniforms = "";

    // Helper to add a line to the shader respecting the current indentation.
    function line(str = "") {
      wgsl += "  ".repeat(indent) + str + "\n";
    }

    // Generate constants for the workgroup size.
    constants += `const kWorkgroupSizeX = ${config.wgsize_x};\n`;
    constants += `const kWorkgroupSizeY = ${config.wgsize_y};\n`;

    // Generate constants for the tile size if necessary.
    if (config.tilesize_x > 1) {
      constants += `const kTileWidth = ${config.tilesize_x};\n`;
    }
    if (config.tilesize_y > 1) {
      constants += `const kTileHeight = ${config.tilesize_y};\n`;
    }
    let tilesize = "";
    if (config.tilesize_x > 1 || config.tilesize_y > 1) {
      const width = config.tilesize_x > 1 ? "kTileWidth" : "1";
      const height = config.tilesize_y > 1 ? "kTileHeight" : "1";
      tilesize = `kTileSize`;
      constants += `const ${tilesize} = vec2(${width}, ${height});\n`;
    }

    // Generate the uniform struct members and the expressions for the filter parameters.
    let uniform_members = "";
    let inv_sigma_domain_sq_expr;
    let inv_sigma_range_sq_expr;
    let radius_expr;
    let width_expr;
    let height_expr;
    if (config.spatial_coeffs === "inline") {
      if (config.const_sigma_domain) {
        constants += `const kInverseSigmaDomainSquared = ${
          -0.5 / (this.sigma_domain * this.sigma_domain)
        };\n`;
        inv_sigma_domain_sq_expr = "kInverseSigmaDomainSquared";
      } else {
        uniform_members += `\n  inv_sigma_domain_sq: f32,`;
        inv_sigma_domain_sq_expr = "params.inv_sigma_domain_sq";
      }
    }
    if (config.const_sigma_range) {
      constants += `const kInverseSigmaRangeSquared = ${
        -0.5 / (this.sigma_range * this.sigma_range)
      };\n`;
      inv_sigma_range_sq_expr = "kInverseSigmaRangeSquared";
    } else {
      uniform_members += `\n  inv_sigma_range_sq: f32,`;
      inv_sigma_range_sq_expr = "params.inv_sigma_range_sq";
    }
    if (config.const_radius) {
      constants += `const kRadius = ${this.radius};\n`;
      radius_expr = "kRadius";
    } else {
      uniform_members += `\n  radius: i32,`;
      radius_expr = "params.radius";
    }
    if (config.const_width) {
      constants += `const kWidth = ${this.width};\n`;
      width_expr = "kWidth";
    } else {
      uniform_members += `\n  width: u32,`;
      width_expr = "params.width";
    }
    if (config.const_height) {
      constants += `const kHeight = ${this.height};\n`;
      height_expr = "kHeight";
    } else {
      uniform_members += `\n  height: u32,`;
      height_expr = "params.height";
    }

    // Emit the uniform struct and variable if there is at least one member.
    if (uniform_members) {
      structures += `struct Parameters {${uniform_members}
}\n`;
      uniforms += "@group(1) @binding(0) var<uniform> params: Parameters;\n";
    }

    // Generate and emit the spatial coefficient LUT if enabled.
    spatial_coeff_lut = new Float32Array((this.radius + 1) * (this.radius + 1));
    for (let j = 0; j < this.radius + 1; j++) {
      for (let i = 0; i < this.radius + 1; i++) {
        let norm = (i * i + j * j) / (this.sigma_domain * this.sigma_domain);
        spatial_coeff_lut[i + j * (this.radius + 1)] = -0.5 * norm;
      }
    }
    if (config.spatial_coeffs === "lut_uniform") {
      const lut_type = `array<vec4f, ${spatial_coeff_lut.length}>`;
      uniforms += `@group(1) @binding(1) var<uniform> spatial_coeff_lut : ${lut_type};
`;
    } else if (config.spatial_coeffs === "lut_const") {
      constants += `const kSpatialCoeffLUT = array<f32, ${spatial_coeff_lut.length}>(`;
      for (let j = 0; j < this.radius + 1; j++) {
        constants += `\n  `;
        for (let i = 0; i < this.radius + 1; i++) {
          constants += `${spatial_coeff_lut[i + j * (this.radius + 1)]}f, `;
        }
      }
      constants += `\n);
`;
    }

    line(`// Constants.\n${constants}`);
    if (structures) {
      line(`// Structures.\n${structures}`);
    }
    if (uniforms) {
      line(`// Uniforms.\n${uniforms}`);
    }

    // Emit the global resources.
    line(`// Inputs and outputs.`);
    line(`@group(0) @binding(0) var input: texture_2d<f32>;`);
    line(`@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;`);
    if (config.input_type === "image_sample") {
      line(`@group(0) @binding(2) var input_sampler: sampler;`);
    }

    // Emit storage for prefetched data if enabled.
    if (config.prefetch === "workgroup") {
      if (!config.const_radius) {
        return "Error: prefetching requires a constant radius.";
      }
      line();
      line(`// Prefetch storage.`);
      line(
        `const kPrefetchWidth = kWorkgroupSizeX${
          config.tilesize_x > 1 ? " * kTileWidth" : ""
        } + 2*${radius_expr};`
      );
      line(
        `const kPrefetchHeight = kWorkgroupSizeY${
          config.tilesize_y > 1 ? " * kTileHeight" : ""
        } + 2*${radius_expr};`
      );
      line(`var<workgroup> prefetch_data: array<vec4f, kPrefetchWidth * kPrefetchHeight>;`);
    }

    // Emit the entry point header.
    line();
    line(`// Entry point.`);
    line(`@compute @workgroup_size(kWorkgroupSizeX, kWorkgroupSizeY)`);
    line(`fn main(@builtin(global_invocation_id) gid: vec3<u32>,`);
    line(`        @builtin(local_invocation_id)  lid: vec3<u32>) {`);
    indent++;
    line(`let step = vec2f(1.f / f32(${width_expr}), 1.f / f32(${height_expr}));`);

    // Prefetch all of the data required by the workgroup if prefetching is enabled.
    if (config.prefetch === "workgroup") {
      line();
      line(`// Prefetch the required data to workgroup storage.`);
      line(
        `let prefetch_base = vec2i(gid.xy - lid.xy)${
          tilesize ? ` * ${tilesize} ` : ""
        } - ${radius_expr};`
      );
      line(`for (var j = i32(lid.y); j < kPrefetchHeight; j += kWorkgroupSizeY) {`);
      indent++;
      line(`for (var i = i32(lid.x); i < kPrefetchWidth; i += kWorkgroupSizeX) {`);
      indent++;
      if (config.input_type === "image_sample") {
        line(`let coord = (vec2f(prefetch_base + vec2(i, j)) + vec2(0.5, 0.5)) * step;`);
        line(`let pixel = textureSampleLevel(input, input_sampler, coord, 0);`);
      } else {
        line(`let coord = prefetch_base + vec2(i, j);`);
        line(`let pixel = textureLoad(input, coord, 0);`);
      }
      line(`prefetch_data[i + j*kPrefetchWidth] = pixel;`);
      indent--;
      line(`}`);
      indent--;
      line(`}`);
      line(`workgroupBarrier();`);
    }

    line();

    // Emit the tile loops if necessary.
    let tx = "0";
    let ty = "0";
    if (config.tilesize_y > 1) {
      ty = "ty";
      line(`for (var ty = 0u; ty < kTileHeight; ty++) {`);
      indent++;
    }
    if (config.tilesize_x > 1) {
      tx = "tx";
      line(`for (var tx = 0u; tx < kTileWidth; tx++) {`);
      indent++;
    }

    // Load the center pixel.
    line(`let center = gid.xy${tilesize ? ` * ${tilesize} + vec2(${tx}, ${ty})` : ""};`);
    if (config.prefetch === "workgroup") {
      line(`let px = lid.x${config.tilesize_x > 1 ? `*kTileWidth + tx` : ""} + ${radius_expr};`);
      line(`let py = lid.y${config.tilesize_y > 1 ? `*kTileHeight + ty` : ""} + ${radius_expr};`);
      line(`let center_value = prefetch_data[px + py*kPrefetchWidth];`);
    } else {
      if (config.input_type === "image_sample") {
        line(`let center_norm = (vec2f(center) + vec2(0.5, 0.5)) * step;`);
        line(`let center_value = textureSampleLevel(input, input_sampler, center_norm, 0);`);
      } else {
        line(`let center_value = textureLoad(input, center, 0);`);
      }
    }

    line();
    line(`var coeff = 0.f;`);
    line(`var sum = vec4f();`);

    // Emit the main filter loop.
    line(`for (var j = -${radius_expr}; j <= ${radius_expr}; j++) {`);
    indent++;
    line(`for (var i = -${radius_expr}; i <= ${radius_expr}; i++) {`);
    indent++;
    line(`var weight = 0.f;`);
    line();

    // Load the pixel from either the texture or the prefetch store.
    if (config.prefetch === "workgroup") {
      line(
        `let px = i32(lid.x${
          config.tilesize_x > 1 ? `*kTileWidth + tx` : ""
        }) + i + ${radius_expr};`
      );
      line(
        `let py = i32(lid.y${
          config.tilesize_y > 1 ? `*kTileHeight + ty` : ""
        }) + j + ${radius_expr};`
      );
      line(`let pixel = prefetch_data[px + py*kPrefetchWidth];`);
    } else {
      if (config.input_type === "image_sample") {
        line(`let coord = center_norm + (vec2(f32(i), f32(j)) * step);`);
        line(`let pixel = textureSampleLevel(input, input_sampler, coord, 0);`);
      } else {
        line(`let pixel = textureLoad(input, vec2i(center) + vec2(i, j), 0);`);
      }
    }

    // Emit the spatial coefficient calculation.
    line();
    if (config.spatial_coeffs === "inline") {
      line(`weight   = (f32(i*i) + f32(j*j)) * ${inv_sigma_domain_sq_expr};`);
    } else if (config.spatial_coeffs === "lut_uniform") {
      line(`weight   = spatial_coeff_lut[abs(i) + abs(j)*(${radius_expr} + 1)].x;`);
    } else if (config.spatial_coeffs === "lut_const") {
      line(`weight   = kSpatialCoeffLUT[abs(i) + abs(j)*(${radius_expr} + 1)];`);
    }

    // Emit the radiometric difference calculation.
    line();
    line(`let diff = pixel.xyz - center_value.xyz;`);
    line(`weight  += dot(diff, diff) * ${inv_sigma_range_sq_expr};`);
    line();

    // Finalize the weight and accumulate into the coefficient and sum.
    line(`weight   = exp(weight);`);
    line(`coeff   += weight;`);
    line(`sum     += weight * pixel;`);
    indent--;
    line(`}`);
    indent--;
    line(`}`);

    // Emit the predicated store for the result.
    line();
    line(`let result = vec4(sum.xyz / coeff, center_value.w);`);
    line(`if (all(center < vec2(${width_expr}, ${height_expr}))) {`);
    line(`  textureStore(output, center, result);`);
    line(`}`);
    indent--;
    line(`}`);

    // End the tile loops if necessary.
    if (config.tilesize_x > 1) {
      indent--;
      line(`}`);
    }
    if (config.tilesize_y > 1) {
      indent--;
      line(`}`);
    }

    return wgsl;
  }

  /// Generate the reference result on the CPU.
  async GenerateReferenceResult() {
    if (this.reference_data) {
      return;
    }

    this.UpdateStatus("Generating reference result...");

    const row_stride = Math.floor((this.width + 255) / 256) * 256;

    // Create a staging buffer for copying the image to the CPU.
    const buffer = this.device.createBuffer({
      size: row_stride * this.height * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Copy the input image to the staging buffer.
    const commands = this.device.createCommandEncoder();
    commands.copyTextureToBuffer(
      { texture: this.input_image_staging },
      { buffer, bytesPerRow: row_stride * 4 },
      {
        width: this.width,
        height: this.height,
      }
    );
    this.device.queue.submit([commands.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const input_data = new Uint8Array(buffer.getMappedRange());

    // Generate the reference output.
    this.reference_data = new Uint8Array(this.width * this.height * 4);
    const inv_255 = 1 / 255.0;
    const inv_sigma_domain_sq = -0.5 / (this.sigma_domain * this.sigma_domain);
    const inv_sigma_range_sq = -0.5 / (this.sigma_range * this.sigma_range);
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        const center_x = input_data[(x + y * row_stride) * 4 + 0] * inv_255;
        const center_y = input_data[(x + y * row_stride) * 4 + 1] * inv_255;
        const center_z = input_data[(x + y * row_stride) * 4 + 2] * inv_255;
        const center_w = input_data[(x + y * row_stride) * 4 + 3];

        let coeff = 0.0;
        let sum = [0, 0, 0];
        for (let j = -this.radius; j <= this.radius; j++) {
          for (let i = -this.radius; i <= this.radius; i++) {
            let xi = Math.min(Math.max(x + i, 0), this.width - 1);
            let yj = Math.min(Math.max(y + j, 0), this.height - 1);
            let pixel_x = input_data[(xi + yj * row_stride) * 4 + 0] * inv_255;
            let pixel_y = input_data[(xi + yj * row_stride) * 4 + 1] * inv_255;
            let pixel_z = input_data[(xi + yj * row_stride) * 4 + 2] * inv_255;

            let weight = (i * i + j * j) * inv_sigma_domain_sq;

            let dist_x = pixel_x - center_x;
            let dist_y = pixel_y - center_y;
            let dist_z = pixel_z - center_z;
            weight += (dist_x * dist_x + dist_y * dist_y + dist_z * dist_z) * inv_sigma_range_sq;

            weight = Math.exp(weight);
            coeff += weight;
            sum[0] += weight * pixel_x;
            sum[1] += weight * pixel_y;
            sum[2] += weight * pixel_z;
          }
        }
        this.reference_data[(x + y * this.width) * 4 + 0] = (sum[0] / coeff) * 255.0;
        this.reference_data[(x + y * this.width) * 4 + 1] = (sum[1] / coeff) * 255.0;
        this.reference_data[(x + y * this.width) * 4 + 2] = (sum[2] / coeff) * 255.0;
        this.reference_data[(x + y * this.width) * 4 + 3] = center_w;
      }
    }

    buffer.unmap();

    this.UpdateStatus("Finished generating reference result.");
  }

  /// Verify a result against the reference result.
  async VerifyResult(params: {
    output: GPUTexture;
    config: ShaderConfig;
    quick?: boolean;
  }): Promise<boolean> {
    this.GenerateReferenceResult();

    this.UpdateStatus("Verifying result...");

    const row_stride = Math.floor((this.width + 255) / 256) * 256;

    // Create a staging buffer for copying the image to the CPU.
    const buffer = this.device.createBuffer({
      size: row_stride * this.height * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Copy the output image to the staging buffer.
    const commands = this.device.createCommandEncoder();
    commands.copyTextureToBuffer(
      { texture: params.output },
      { buffer, bytesPerRow: row_stride * 4 },
      {
        width: this.width,
        height: this.height,
      }
    );
    this.device.queue.submit([commands.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const result_data = new Uint8Array(buffer.getMappedRange());

    // Check for errors and generate the diff map.
    let num_errors = 0;
    let max_error = 0;
    this.diff_data = new Uint8Array(this.width * this.height * 4);
    const x_region = params.quick
      ? 2 * (params.config.wgsize_x * params.config.tilesize_x) + this.radius
      : this.width;
    const y_region = params.quick
      ? 2 * (params.config.wgsize_y * params.config.tilesize_y) + this.radius
      : this.height;
    for (let y = 0; y < this.height; y++) {
      if (y >= y_region && y < this.height - y_region) {
        continue;
      }
      for (let x = 0; x < this.width; x++) {
        if (x >= x_region && x < this.width - x_region) {
          continue;
        }

        // Use green for a match.
        this.diff_data[(x + y * this.width) * 4 + 0] = 0;
        this.diff_data[(x + y * this.width) * 4 + 1] = 255;
        this.diff_data[(x + y * this.width) * 4 + 2] = 0;
        this.diff_data[(x + y * this.width) * 4 + 3] = 255;

        let has_error = false;
        for (let c = 0; c < 4; c++) {
          const result = result_data[(x + y * row_stride) * 4 + c];
          const reference = this.reference_data[(x + y * this.width) * 4 + c];
          const diff = Math.abs(result - reference);
          if (diff > 1) {
            // Use red for large errors, orange for smaller errors.
            if (diff > 20) {
              this.diff_data[(x + y * this.width) * 4 + 0] = 255;
              this.diff_data[(x + y * this.width) * 4 + 1] = 0;
            } else {
              this.diff_data[(x + y * this.width) * 4 + 0] = 255;
              this.diff_data[(x + y * this.width) * 4 + 1] = 165;
            }
            max_error = Math.max(max_error, diff);
            if (num_errors < 10) {
              console.log(`error at ${x},${y},${c}: ${result} != ${reference}`);
            }
            if (!has_error) {
              num_errors++;
              has_error = true;
            }
          }
        }
      }
    }
    buffer.unmap();

    if (num_errors) {
      this.UpdateStatus(`${num_errors} errors found (maxdiff=${max_error}).`, "#FF0000");
    } else {
      this.UpdateStatus("Verification succeeded.");
    }

    return num_errors == 0;
  }
}
