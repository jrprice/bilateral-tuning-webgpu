// The filter parameters.
// TODO: Make the sigma_domain and sigma_range parameters configurable.
let sigma_domain = 3.0;
let sigma_range = 0.2;
let radius = 2;
// The shader parameters.
let wgsize_x = 8;
let wgsize_y = 8;
let const_sigma_domain;
let const_sigma_range;
let const_radius;
let const_width;
let const_height;
let prefetch = "none";
let spatial_coeffs = "inline";
// The input image data.
let width;
let height;
// The spatial coefficient LUT.
let spatial_coeff_lut = null;
// The reference result.
let reference_data = null;
// WebGPU objects.
let adapter;
let device;
let input_image_staging;
function SetStatus(str) {
    document.getElementById("status").textContent = str;
}
/// Initialize the main WebGPU objects.
async function InitWebGPU() {
    // Initialize the WebGPU device and queue.
    SetStatus("Initializing...");
    const powerpref = document.getElementById("powerpref").value;
    adapter = await navigator.gpu.requestAdapter({
        powerPreference: powerpref,
    });
    device = await adapter.requestDevice();
}
/// Get the texture for the canvas called `id`.
function GetCanvasTexture(id) {
    const canvas = document.getElementById(id);
    canvas.width = width;
    canvas.height = height;
    const canvas_context = canvas.getContext("webgpu");
    canvas_context.configure({
        device,
        format: navigator.gpu.getPreferredCanvasFormat(),
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    return canvas_context.getCurrentTexture();
}
/// Load the currently selected input image and display it to the input canvas.
async function LoadInputImage() {
    SetStatus("Loading input image...");
    // Load the input image from file.
    const image_selector = document.getElementById("image_file");
    const filename = image_selector.selectedOptions[0].value;
    const response = await fetch(`dist/${filename}.jpg`);
    const blob = await response.blob();
    const input_bitmap = await createImageBitmap(blob);
    width = input_bitmap.width;
    height = input_bitmap.height;
    // Copy the input image to a staging texture.
    input_image_staging = device.createTexture({
        format: "rgba8unorm",
        size: { width, height },
        usage: GPUTextureUsage.COPY_SRC |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT |
            GPUTextureUsage.TEXTURE_BINDING,
    });
    device.queue.copyExternalImageToTexture({ source: input_bitmap }, { texture: input_image_staging }, { width, height });
    DisplayTexture(input_image_staging, "input_canvas");
    // Reconfigure the other canvases to clear and resize them.
    GetCanvasTexture("output_canvas");
    // Clear the reference canvases and data.
    function ClearCanvas(id) {
        const canvas = document.getElementById(id);
        canvas.width = width;
        canvas.height = height;
        canvas.getContext("2d").clearRect(0, 0, width, height);
    }
    ClearCanvas("reference_canvas");
    ClearCanvas("diff_canvas");
    reference_data = null;
    SetStatus("Ready.");
}
/// Run the benchmark.
const Run = async () => {
    SetStatus("Setting up...");
    // Cycle through different input images to reduce the likelihood of caching.
    const kNumInputImages = 3;
    // Create the input and output textures.
    const inputs = [];
    const commands = device.createCommandEncoder();
    for (let i = 0; i < kNumInputImages; i++) {
        inputs.push(device.createTexture({
            size: { width, height },
            format: "rgba8unorm",
            usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
        }));
        // Copy the input image to the input texture.
        commands.copyTextureToTexture({ texture: input_image_staging }, { texture: inputs[i] }, {
            width,
            height,
        });
    }
    const output = device.createTexture({
        size: { width, height },
        format: "rgba8unorm",
        usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    // Wait for copies to complete.
    device.queue.submit([commands.finish()]);
    // Set up the filter parameters.
    const parameters = device.createBuffer({
        size: 20,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true,
    });
    const param_values = parameters.getMappedRange();
    const param_values_f32 = new Float32Array(param_values);
    const param_values_u32 = new Uint32Array(param_values);
    // Set values for any parameters that are not being embedded as constants.
    let uniform_member_index = 0;
    if (!const_sigma_domain) {
        param_values_f32[uniform_member_index++] = -0.5 / (sigma_domain * sigma_domain);
    }
    if (!const_sigma_range) {
        param_values_f32[uniform_member_index++] = -0.5 / (sigma_range * sigma_range);
    }
    if (!const_radius) {
        param_values_u32[uniform_member_index++] = radius;
    }
    if (!const_width) {
        param_values_u32[uniform_member_index++] = width;
    }
    if (!const_height) {
        param_values_u32[uniform_member_index++] = height;
    }
    parameters.unmap();
    // Generate the shader and create the compute pipeline.
    const module = device.createShaderModule({ code: GenerateShader() });
    const pipeline = device.createComputePipeline({
        compute: { module, entryPoint: "main" },
        layout: "auto",
    });
    // Create a bind group for group index 0 for each input image.
    const bind_group_0 = [];
    for (let i = 0; i < kNumInputImages; i++) {
        bind_group_0.push(device.createBindGroup({
            entries: [
                { binding: 0, resource: inputs[i].createView() },
                { binding: 1, resource: output.createView() },
                {
                    binding: 2,
                    resource: device.createSampler({
                        addressModeU: "clamp-to-edge",
                        addressModeV: "clamp-to-edge",
                        minFilter: "nearest",
                        magFilter: "nearest",
                    }),
                },
            ],
            layout: pipeline.getBindGroupLayout(0),
        }));
    }
    // Create a uniform buffer for the spatial coefficient LUT if necessary.
    let spatial_coeff_lut_buffer = null;
    if (spatial_coeffs === "lut_uniform") {
        let buffer_size = spatial_coeff_lut.length * 16;
        spatial_coeff_lut_buffer = device.createBuffer({
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
        bind_group_1 = device.createBindGroup({
            entries: bind_group_1_entries,
            layout: pipeline.getBindGroupLayout(1),
        });
    }
    // Determine the number of workgroups.
    const group_count_x = Math.floor((width + wgsize_x - 1) / wgsize_x);
    const group_count_y = Math.floor((height + wgsize_y - 1) / wgsize_y);
    // Helper to enqueue `n` back-to-back runs of the shader.
    function Enqueue(n) {
        const commands = device.createCommandEncoder();
        const pass = commands.beginComputePass();
        pass.setPipeline(pipeline);
        if (bind_group_1) {
            // Only set the bind group for the uniform parameters if it was created.
            pass.setBindGroup(1, bind_group_1);
        }
        for (let i = 0; i < n; i++) {
            pass.setBindGroup(0, bind_group_0[n % kNumInputImages]);
            pass.dispatchWorkgroups(group_count_x, group_count_y);
        }
        pass.end();
        device.queue.submit([commands.finish()]);
    }
    // Warm up run.
    Enqueue(1);
    await device.queue.onSubmittedWorkDone();
    // Timed runs.
    SetStatus("Running...");
    const itrs = +document.getElementById("iterations").value;
    const start = performance.now();
    Enqueue(itrs);
    await device.queue.onSubmittedWorkDone();
    const end = performance.now();
    const elapsed = end - start;
    const fps = (itrs / elapsed) * 1000;
    const perf_str = `Elapsed time: ${elapsed.toFixed(2)} ms (${fps.toFixed(2)} frames/second)`;
    document.getElementById("runtime").textContent = perf_str;
    DisplayTexture(output, "output_canvas");
    VerifyResult(output);
};
/// Generate the WGSL shader.
function GenerateShader() {
    let wgsl = "";
    let constants = "";
    let structures = "";
    let uniforms = "";
    // Generate constants for the workgroup size.
    constants += `const kWorkgroupSizeX = ${wgsize_x};\n`;
    constants += `const kWorkgroupSizeY = ${wgsize_y};\n`;
    // Generate the uniform struct members and the expressions for the filter parameters.
    let uniform_members = "";
    let inv_sigma_domain_sq_expr;
    let inv_sigma_range_sq_expr;
    let radius_expr;
    let width_expr;
    let height_expr;
    if (const_sigma_domain) {
        constants += `const kInverseSigmaDomainSquared = ${-0.5 / (sigma_domain * sigma_domain)};\n`;
        inv_sigma_domain_sq_expr = "kInverseSigmaDomainSquared";
    }
    else {
        uniform_members += `\n  inv_sigma_domain_sq: f32,`;
        inv_sigma_domain_sq_expr = "params.inv_sigma_domain_sq";
    }
    if (const_sigma_range) {
        constants += `const kInverseSigmaRangeSquared = ${-0.5 / (sigma_range * sigma_range)};\n`;
        inv_sigma_range_sq_expr = "kInverseSigmaRangeSquared";
    }
    else {
        uniform_members += `\n  inv_sigma_range_sq: f32,`;
        inv_sigma_range_sq_expr = "params.inv_sigma_range_sq";
    }
    if (const_radius) {
        constants += `const kRadius = ${radius};\n`;
        radius_expr = "kRadius";
    }
    else {
        uniform_members += `\n  radius: i32,`;
        radius_expr = "params.radius";
    }
    if (const_width) {
        constants += `const kWidth = ${width};\n`;
        width_expr = "kWidth";
    }
    else {
        uniform_members += `\n  width: u32,`;
        width_expr = "params.width";
    }
    if (const_height) {
        constants += `const kHeight = ${height};\n`;
        height_expr = "kHeight";
    }
    else {
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
    spatial_coeff_lut = new Float32Array((radius + 1) * (radius + 1));
    for (let j = 0; j < radius + 1; j++) {
        for (let i = 0; i < radius + 1; i++) {
            let norm = (i * i + j * j) / (sigma_domain * sigma_domain);
            spatial_coeff_lut[i + j * (radius + 1)] = -0.5 * norm;
        }
    }
    if (spatial_coeffs === "lut_uniform") {
        const lut_type = `array<vec4f, ${spatial_coeff_lut.length}>`;
        uniforms += `@group(1) @binding(1) var<uniform> spatial_coeff_lut : ${lut_type};
`;
    }
    else if (spatial_coeffs === "lut_const") {
        constants += `const kSpatialCoeffLUT = array<f32, ${spatial_coeff_lut.length}>(`;
        for (let j = 0; j < radius + 1; j++) {
            constants += `\n  `;
            for (let i = 0; i < radius + 1; i++) {
                constants += `${spatial_coeff_lut[i + j * (radius + 1)]}f, `;
            }
        }
        constants += `\n);
`;
    }
    wgsl += `// Constants.\n${constants}\n`;
    if (structures) {
        wgsl += `// Structures.\n${structures}\n`;
    }
    if (uniforms) {
        wgsl += `// Uniforms.\n${uniforms}\n`;
    }
    // Emit the global resources.
    wgsl += `// Inputs and outputs.
@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var input_sampler: sampler;
`;
    // Emit storage for prefetched data if enabled.
    if (prefetch === "workgroup") {
        if (!const_radius) {
            return "Error: prefetching requires a constant radius.";
        }
        wgsl += `\n// Prefetch storage.
const kPrefetchWidth = kWorkgroupSizeX + 2*${radius_expr};
const kPrefetchHeight = kWorkgroupSizeY + 2*${radius_expr};
var<workgroup> prefetch_data: array<vec4f, kPrefetchWidth * kPrefetchHeight>;
`;
    }
    // Emit the entry point header.
    wgsl += `\n// Entry point.
@compute @workgroup_size(kWorkgroupSizeX, kWorkgroupSizeY)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
  let step = vec2f(1.f / f32(${width_expr}), 1.f / f32(${height_expr}));
`;
    // Emit code to prefetch required data if prefetching is enabled, and load the center pixel.
    if (prefetch === "workgroup") {
        wgsl += `
  // Prefetch the required data to workgroup storage.
  let prefetch_base = vec2i(gid.xy - lid.xy) - ${radius_expr};
  for (var j = i32(lid.y); j < kPrefetchHeight; j += kWorkgroupSizeY) {
    for (var i = i32(lid.x); i < kPrefetchWidth; i += kWorkgroupSizeX) {
      let coord = (vec2f(prefetch_base + vec2i(i, j)) + vec2f(0.5, 0.5)) * step;
      prefetch_data[i + j*kPrefetchWidth] = textureSampleLevel(input, input_sampler, coord, 0);
    }
  }
  workgroupBarrier();

  let center_value = prefetch_data[lid.x+${radius_expr} + (lid.y+${radius_expr})*kPrefetchWidth];
`;
    }
    else {
        wgsl += `
  let center = (vec2f(gid.xy) + vec2f(0.5, 0.5)) * step;
  let center_value = textureSampleLevel(input, input_sampler, center, 0);
`;
    }
    // Emit the body of the shader.
    wgsl += `
  var coeff = 0.f;
  var sum = vec4f();
  for (var j = -${radius_expr}; j <= ${radius_expr}; j++) {
    for (var i = -${radius_expr}; i <= ${radius_expr}; i++) {
      var weight = 0.f;
`;
    // Load the pixel from either the texture or the prefetch store.
    if (prefetch === "workgroup") {
        wgsl += `
      let x = i32(lid.x) + i + ${radius_expr};
      let y = i32(lid.y) + j + ${radius_expr};
      let pixel = prefetch_data[x + y*kPrefetchWidth];
`;
    }
    else {
        wgsl += `
      let coord = center + (vec2f(f32(i), f32(j)) * step);
      let pixel = textureSampleLevel(input, input_sampler, coord, 0);
`;
    }
    // Emit the spatial coefficient calculation.
    if (spatial_coeffs === "inline") {
        wgsl += `
      weight   = (f32(i*i) + f32(j*j)) * ${inv_sigma_domain_sq_expr};
`;
    }
    else if (spatial_coeffs === "lut_uniform") {
        wgsl += `
      weight   = spatial_coeff_lut[abs(i) + abs(j)*(${radius_expr} + 1)].x;
`;
    }
    else if (spatial_coeffs === "lut_const") {
        wgsl += `
      weight   = kSpatialCoeffLUT[abs(i) + abs(j)*(${radius_expr} + 1)];
`;
    }
    // Emit the weight calculations.
    wgsl += `
      let diff = pixel.xyz - center_value.xyz;
      weight  += dot(diff, diff) * ${inv_sigma_range_sq_expr};

      weight   = exp(weight);
      coeff   += weight;
      sum     += weight * pixel;
    }
  }
`;
    // Emit the predicated store for the result.
    wgsl += `
  let result = vec4f(sum.xyz / coeff, center_value.w);
  if (all(gid.xy < vec2u(${width_expr}, ${height_expr}))) {
    textureStore(output, gid.xy, result);
  }
}`;
    return wgsl;
}
/// Update and display the WGSL shader.
function UpdateShader() {
    const shader_display = document.getElementById("shader");
    shader_display.style.width = `0px`;
    shader_display.style.height = `0px`;
    shader_display.textContent = GenerateShader();
    shader_display.style.width = `${shader_display.scrollWidth}px`;
    shader_display.style.height = `${shader_display.scrollHeight}px`;
}
/// Display a texture to a canvas.
function DisplayTexture(texture, canvas_id) {
    const module = device.createShaderModule({
        code: `
@vertex
fn vert_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  const kVertices = array(
    vec2f(-1, -1),
    vec2f(-1,  1),
    vec2f( 1, -1),
    vec2f(-1,  1),
    vec2f( 1, -1),
    vec2f( 1,  1),
  );
  return vec4f(kVertices[idx], 0, 1);
}

@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment
fn frag_main(@builtin(position) position: vec4f) -> @location(0) vec4f {
  return textureSampleLevel(image, s, position.xy / vec2f(${width}, ${height}), 0);
}
  `,
    });
    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { entryPoint: "vert_main", module },
        fragment: {
            entryPoint: "frag_main",
            module,
            targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
        },
    });
    const commands = device.createCommandEncoder();
    const pass = commands.beginRenderPass({
        colorAttachments: [
            {
                view: GetCanvasTexture(canvas_id).createView(),
                loadOp: "clear",
                storeOp: "store",
            },
        ],
    });
    pass.setBindGroup(0, device.createBindGroup({
        entries: [
            { binding: 0, resource: texture.createView() },
            { binding: 1, resource: device.createSampler() },
        ],
        layout: pipeline.getBindGroupLayout(0),
    }));
    pass.setPipeline(pipeline);
    pass.draw(6);
    pass.end();
    device.queue.submit([commands.finish()]);
}
/// Display image data to a canvas from a Uint8Array.
function DisplayImageData(data, canvas_id) {
    const imgdata = new ImageData(new Uint8ClampedArray(data), width, height);
    const canvas = document.getElementById(canvas_id);
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").putImageData(imgdata, 0, 0);
}
/// Generate the reference result on the CPU.
async function GenerateReferenceResult() {
    if (reference_data) {
        return;
    }
    SetStatus("Generating reference result...");
    const row_stride = Math.floor((width + 255) / 256) * 256;
    // Create a staging buffer for copying the image to the CPU.
    const buffer = device.createBuffer({
        size: row_stride * height * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    // Copy the input image to the staging buffer.
    const commands = device.createCommandEncoder();
    commands.copyTextureToBuffer({ texture: input_image_staging }, { buffer, bytesPerRow: row_stride * 4 }, {
        width,
        height,
    });
    device.queue.submit([commands.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const input_data = new Uint8Array(buffer.getMappedRange());
    // Generate the reference output.
    reference_data = new Uint8Array(width * height * 4);
    const inv_255 = 1 / 255.0;
    const inv_sigma_domain_sq = -0.5 / (sigma_domain * sigma_domain);
    const inv_sigma_range_sq = -0.5 / (sigma_range * sigma_range);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const center_x = input_data[(x + y * row_stride) * 4 + 0] * inv_255;
            const center_y = input_data[(x + y * row_stride) * 4 + 1] * inv_255;
            const center_z = input_data[(x + y * row_stride) * 4 + 2] * inv_255;
            const center_w = input_data[(x + y * row_stride) * 4 + 3];
            let coeff = 0.0;
            let sum = [0, 0, 0];
            for (let j = -radius; j <= radius; j++) {
                for (let i = -radius; i <= radius; i++) {
                    let xi = Math.min(Math.max(x + i, 0), width - 1);
                    let yj = Math.min(Math.max(y + j, 0), height - 1);
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
            reference_data[(x + y * width) * 4 + 0] = (sum[0] / coeff) * 255.0;
            reference_data[(x + y * width) * 4 + 1] = (sum[1] / coeff) * 255.0;
            reference_data[(x + y * width) * 4 + 2] = (sum[2] / coeff) * 255.0;
            reference_data[(x + y * width) * 4 + 3] = center_w;
        }
    }
    buffer.unmap();
    DisplayImageData(reference_data, "reference_canvas");
}
/// Verify a result against the reference result.
async function VerifyResult(output) {
    // Generate the reference result.
    await GenerateReferenceResult();
    SetStatus("Verifying result...");
    const row_stride = Math.floor((width + 255) / 256) * 256;
    // Create a staging buffer for copying the image to the CPU.
    const buffer = device.createBuffer({
        size: row_stride * height * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    // Copy the output image to the staging buffer.
    const commands = device.createCommandEncoder();
    commands.copyTextureToBuffer({ texture: output }, { buffer, bytesPerRow: row_stride * 4 }, {
        width,
        height,
    });
    device.queue.submit([commands.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const result_data = new Uint8Array(buffer.getMappedRange());
    // Check for errors and generate the diff map.
    let num_errors = 0;
    let max_error = 0;
    const diff_data = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            // Use green for a match.
            diff_data[(x + y * width) * 4 + 0] = 0;
            diff_data[(x + y * width) * 4 + 1] = 255;
            diff_data[(x + y * width) * 4 + 2] = 0;
            diff_data[(x + y * width) * 4 + 3] = 255;
            let has_error = false;
            for (let c = 0; c < 4; c++) {
                const result = result_data[(x + y * row_stride) * 4 + c];
                const reference = reference_data[(x + y * width) * 4 + c];
                const diff = Math.abs(result - reference);
                if (diff > 1) {
                    // Use red for large errors, orange for smaller errors.
                    if (diff > 20) {
                        diff_data[(x + y * width) * 4 + 0] = 255;
                        diff_data[(x + y * width) * 4 + 1] = 0;
                    }
                    else {
                        diff_data[(x + y * width) * 4 + 0] = 255;
                        diff_data[(x + y * width) * 4 + 1] = 165;
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
        SetStatus(`${num_errors} errors found (maxdiff=${max_error}).`);
    }
    else {
        SetStatus("Verification succeeded.");
    }
    // Display the image diff.
    DisplayImageData(diff_data, "diff_canvas");
}
// Initialize WebGPU.
await InitWebGPU();
// Display the default shader.
UpdateShader();
// Add an event handler for the 'Run' button.
document.querySelector("#run").addEventListener("click", Run);
// Add an event handler for the power preference selector.
document.querySelector("#powerpref").addEventListener("change", () => {
    InitWebGPU();
    LoadInputImage();
});
// Add an event handler for the image selector.
document.querySelector("#image_file").addEventListener("change", () => {
    LoadInputImage();
});
// Add an event handler for the radius selector.
document.querySelector("#radius").addEventListener("change", () => {
    radius = +document.getElementById("radius").value;
    reference_data = null;
    UpdateShader();
});
// Add event handlers for the shader parameter radio buttons.
document.querySelector("#const_sd").addEventListener("change", () => {
    const_sigma_domain = true;
    UpdateShader();
});
document.querySelector("#uniform_sd").addEventListener("change", () => {
    const_sigma_domain = false;
    UpdateShader();
});
document.querySelector("#const_sr").addEventListener("change", () => {
    const_sigma_range = true;
    UpdateShader();
});
document.querySelector("#uniform_sr").addEventListener("change", () => {
    const_sigma_range = false;
    UpdateShader();
});
document.querySelector("#const_radius").addEventListener("change", () => {
    const_radius = true;
    UpdateShader();
});
document.querySelector("#uniform_radius").addEventListener("change", () => {
    const_radius = false;
    UpdateShader();
});
document.querySelector("#const_width").addEventListener("change", () => {
    const_width = true;
    UpdateShader();
});
document.querySelector("#uniform_width").addEventListener("change", () => {
    const_width = false;
    UpdateShader();
});
document.querySelector("#const_height").addEventListener("change", () => {
    const_height = true;
    UpdateShader();
});
document.querySelector("#uniform_height").addEventListener("change", () => {
    const_height = false;
    UpdateShader();
});
// Add event handlers for the shader parameter drop-down menus.
document.querySelector("#wgsize_x").addEventListener("change", () => {
    wgsize_x = +document.getElementById("wgsize_x").value;
    UpdateShader();
});
document.querySelector("#wgsize_y").addEventListener("change", () => {
    wgsize_y = +document.getElementById("wgsize_y").value;
    UpdateShader();
});
document.querySelector("#prefetch").addEventListener("change", () => {
    prefetch = document.getElementById("prefetch").value;
    UpdateShader();
});
document.querySelector("#spatial_coeffs").addEventListener("change", () => {
    spatial_coeffs = document.getElementById("spatial_coeffs").value;
    UpdateShader();
});
// Load the default input image.
LoadInputImage();
export {};
