export {};

// The input image data.
let width: number;
let height: number;
let input_bitmap: ImageBitmap;

// WebGPU objects.
let adapter: GPUAdapter;
let device: GPUDevice;

/// Initialize the main WebGPU objects.
async function InitWebGPU() {
  // Initialize the WebGPU device and queue.
  adapter = await navigator.gpu.requestAdapter();
  device = await adapter.requestDevice();
}

/// Get the texture for the canvas called `id`.
function GetCanvasTexture(id: string) {
  const canvas = <HTMLCanvasElement>document.getElementById(id);
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
  // Load the input image from file.
  const image_selector = <HTMLSelectElement>(
    document.getElementById("image_file")
  );
  const filename = image_selector.selectedOptions[0].value;
  const response = await fetch(`dist/${filename}.jpg`);
  const blob = await response.blob();
  input_bitmap = await createImageBitmap(blob);
  width = input_bitmap.width;
  height = input_bitmap.height;

  // Display the input image to the input canvas.
  let input_canvas = GetCanvasTexture("input_canvas");
  device.queue.copyExternalImageToTexture(
    { source: input_bitmap },
    { texture: input_canvas },
    { width, height }
  );

  // Reconfigure the output canvas to clear it and resize it.
  GetCanvasTexture("output_canvas");
}

/// Run the benchmark.
const Run = async () => {
  // Create the input and output textures.
  const input = device.createTexture({
    size: { width, height },
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const output = device.createTexture({
    size: { width, height },
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING,
  });

  // Copy the input image to the input texture.
  device.queue.copyExternalImageToTexture(
    { source: input_bitmap },
    { texture: input },
    { width, height }
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
  param_values_u32[3] = width;
  param_values_u32[4] = height;
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
      pass.dispatchWorkgroups(width / 16, height / 16);
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

  DisplayResult(output);

  // TODO: Verify the result.
};

/// Generate the WGSL shader.
function GenerateShader(): string {
  return `
struct Parameters {
  inv_sigma_domain: f32,
  inv_sigma_range: f32,
  radius: i32,
  width: u32,
  height: u32,
}

@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var input_sampler: sampler;

@group(1) @binding(0) var<uniform> params: Parameters;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let center = vec2i(gid.xy);
  let center_norm = vec2f(center) / vec2f(f32(params.width), f32(params.height));
  let center_value = textureSampleLevel(input, input_sampler, center_norm, 0);
  let dx = 1.f / f32(params.width);
  let dy = 1.f / f32(params.height);

  var coeff = 0.f;
  var sum = vec4f();
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      var norm = 0.f;
      var weight = 0.f;

      let coord = center_norm + vec2f(f32(i) * dx, f32(j) * dy);
      let pixel = textureSampleLevel(input, input_sampler, coord, 0);

      norm    = sqrt(f32(i*i) + f32(j*j)) * params.inv_sigma_domain;
      weight  = -0.5f * (norm * norm);

      norm    = distance(pixel.xyz, center_value.xyz) * params.inv_sigma_range;
      weight += -0.5f * (norm * norm);

      weight = exp(weight);
      coeff += weight;
      sum   += weight * pixel;
    }
  }

  let result = vec4f(sum.xyz / coeff, center_value.w);
  textureStore(output, center, result);
}`;
}

/// Display the result in the output canvas.
function DisplayResult(result: GPUTexture) {
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
        view: GetCanvasTexture("output_canvas").createView(),
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  pass.setBindGroup(
    0,
    device.createBindGroup({
      entries: [
        { binding: 0, resource: result.createView() },
        { binding: 1, resource: device.createSampler() },
      ],
      layout: pipeline.getBindGroupLayout(0),
    })
  );
  pass.setPipeline(pipeline);
  pass.draw(6);
  pass.end();
  device.queue.submit([commands.finish()]);
}

// Initialize WebGPU.
await InitWebGPU();

// Load the default input image.
LoadInputImage();

// Add an event handler for the 'Run' button.
document.querySelector("#run").addEventListener("click", Run);

// Add an event handler for the image selector.
document.querySelector("#image_file").addEventListener("change", () => {
  LoadInputImage();
});
