// TODO: Make the size configurable.
const kWidth = 4096;
const kHeight = 4096;

// The input image data.
const input_data = new Uint8Array(kWidth * kHeight * 4);

/// Run the benchmark.
const Run = async () => {
  // Initialize the WebGPU device and queue.
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  // Create the input and output textures.
  const input = device.createTexture({
    size: { width: kWidth, height: kHeight },
    format: "rgba8unorm",
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  });
  const output = device.createTexture({
    size: { width: kWidth, height: kHeight },
    format: "rgba8unorm",
    usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING,
  });

  // Copy the input data to the input texture.
  device.queue.writeTexture(
    { texture: input },
    input_data,
    { bytesPerRow: kWidth * 4 },
    { width: kWidth, height: kHeight }
  );

  // Set up the filter parameters.
  const parameters = device.createBuffer({
    size: 20,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  const param_values = parameters.getMappedRange();
  const param_values_f32 = new Float32Array(param_values);
  const param_values_u32 = new Uint32Array(param_values);
  // TODO: Make the parameters configurable.
  param_values_f32[0] = 1.0 / 3.0;
  param_values_f32[1] = 1.0 / 0.2;
  param_values_u32[2] = 2;
  param_values_u32[3] = kWidth;
  param_values_u32[4] = kHeight;
  parameters.unmap();

  // Generate the shader and create the compute pipeline.
  const module = device.createShaderModule({ code: GenerateShader() });
  const pipeline = device.createComputePipeline({
    compute: { module, entryPoint: "main" },
    layout: "auto",
  });

  const bind_group_0 = device.createBindGroup({
    entries: [
      { binding: 0, resource: input.createView() },
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
  });
  const bind_group_1 = device.createBindGroup({
    entries: [{ binding: 0, resource: { buffer: parameters } }],
    layout: pipeline.getBindGroupLayout(1),
  });

  // Helper to enqueue `n` back-to-back runs of the shader.
  function Enqueue(n: number) {
    const commands = device.createCommandEncoder();
    const pass = commands.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind_group_0);
    pass.setBindGroup(1, bind_group_1);
    for (let i = 0; i < n; i++) {
      // TODO: Handle width and height not divisible by the workgroup size.
      pass.dispatchWorkgroups(kWidth / 16, kHeight / 16);
    }
    pass.end();
    device.queue.submit([commands.finish()]);
  }

  // Warm up run.
  Enqueue(1);
  await device.queue.onSubmittedWorkDone();

  // Timed runs.
  const start = performance.now();
  Enqueue(100);
  await device.queue.onSubmittedWorkDone();
  const end = performance.now();
  const elapsed = end - start;
  console.log(`Elapsed time: ${elapsed.toFixed(2)}ms`);

  console.log('Finished.');

  // TODO: Verify the result.
};

/// Generate the WGSL shader.
function GenerateShader(): string {
  return `
struct Parameters {
  inv_sigma_domain : f32,
  inv_sigma_range : f32,
  radius : i32,
  width : u32,
  height : u32,
}

@group(0) @binding(0) var input : texture_2d<f32>;
@group(0) @binding(1) var output : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var input_sampler : sampler;

@group(1) @binding(0) var<uniform> params : Parameters;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let center = vec2i(gid.xy);
  let center_value = textureSampleLevel(input, input_sampler, vec2f(center), 0);

  var coeff = 0.f;
  var sum = vec4f();
  for (var j = -params.radius; j <= params.radius; j++) {
    for (var i = -params.radius; i <= params.radius; i++) {
      var norm : f32;
      var weight : f32;
      let pixel = textureSampleLevel(input, input_sampler, vec2f(center + vec2i(i,j)), 0);

      norm    = sqrt(f32(i*i) + f32(j*j)) * params.inv_sigma_domain;
      weight  = -0.5f * (norm * norm);

      norm    = distance(pixel.xyz, center_value.xyz) * params.inv_sigma_range;
      weight += -0.5f * (norm * norm);

      weight = exp(weight);
      coeff += weight;
      sum   += weight*pixel;
    }
  }

  let result = vec4f(sum.xyz / coeff, center_value.w);
  textureStore(output, center, result);
}`;
}

/// Initialize the input image data.
function InitImage(input_data: Uint8Array) {
  // Generate a random image.
  for (let y = 0; y < kHeight; y++) {
    for (let x = 0; x < kWidth; x++) {
      input_data[(x + y * kWidth) * 4 + 0] = Math.random() * 255;
      input_data[(x + y * kWidth) * 4 + 1] = Math.random() * 255;
      input_data[(x + y * kWidth) * 4 + 2] = Math.random() * 255;
      input_data[(x + y * kWidth) * 4 + 3] = 255;
    }
  }
}

// Initialize the input data.
InitImage(input_data);

// Add an event handler for the "Run" button.
document.querySelector("#run").addEventListener("click", Run);
