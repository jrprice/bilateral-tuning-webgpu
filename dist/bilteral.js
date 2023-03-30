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
// The input image data.
let width;
let height;
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
    adapter = await navigator.gpu.requestAdapter();
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
    // Create the input and output textures.
    const input = device.createTexture({
        size: { width, height },
        format: "rgba8unorm",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    });
    const output = device.createTexture({
        size: { width, height },
        format: "rgba8unorm",
        usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    // Copy the input image to the input texture.
    const commands = device.createCommandEncoder();
    commands.copyTextureToTexture({ texture: input_image_staging }, { texture: input }, {
        width,
        height,
    });
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
        param_values_f32[uniform_member_index++] = 1.0 / sigma_domain;
    }
    if (!const_sigma_range) {
        param_values_f32[uniform_member_index++] = 1.0 / sigma_range;
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
    // Create the bind group for the uniform parameters if necessary.
    let bind_group_1 = null;
    if (uniform_member_index > 0) {
        bind_group_1 = device.createBindGroup({
            entries: [{ binding: 0, resource: { buffer: parameters } }],
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
        pass.setBindGroup(0, bind_group_0);
        if (bind_group_1) {
            // Only set the bind group for the uniform parameters if it was created.
            pass.setBindGroup(1, bind_group_1);
        }
        for (let i = 0; i < n; i++) {
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
    // Generate the uniform struct members and the expressions for the filter parameters.
    let uniform_members = "";
    let inv_sigma_domain_expr;
    let inv_sigma_range_expr;
    let radius_expr;
    let width_expr;
    let height_expr;
    if (const_sigma_domain) {
        inv_sigma_domain_expr = `${1.0 / sigma_domain}`;
    }
    else {
        uniform_members += `\n  inv_sigma_domain: f32,`;
        inv_sigma_domain_expr = "params.inv_sigma_domain";
    }
    if (const_sigma_range) {
        inv_sigma_range_expr = `${1.0 / sigma_range}`;
    }
    else {
        uniform_members += `\n  inv_sigma_range: f32,`;
        inv_sigma_range_expr = "params.inv_sigma_range";
    }
    if (const_radius) {
        radius_expr = `${radius}`;
    }
    else {
        uniform_members += `\n  radius: i32,`;
        radius_expr = "params.radius";
    }
    if (const_width) {
        width_expr = `${width}`;
    }
    else {
        uniform_members += `\n  width: u32,`;
        width_expr = "params.width";
    }
    if (const_height) {
        height_expr = `${height}`;
    }
    else {
        uniform_members += `\n  height: u32,`;
        height_expr = "params.height";
    }
    // Emit the uniform struct and variable if there is at least one member.
    if (uniform_members) {
        wgsl += `struct Parameters {${uniform_members}
}
@group(1) @binding(0) var<uniform> params: Parameters;

`;
    }
    // Emit the global resources.
    wgsl += `@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var input_sampler: sampler;
`;
    // Emit storage for prefetched data if enabled.
    if (prefetch === "workgroup") {
        if (!const_radius) {
            return "Error: prefetching requires a constant radius.";
        }
        wgsl += `
const kPrefetchWidth = wgsize_x + 2*${radius_expr};
const kPrefetchHeight = wgsize_y + 2*${radius_expr};
var<workgroup> prefetch_data: array<vec4f, kPrefetchWidth * kPrefetchHeight>;
`;
    }
    // Emit the entry point header.
    wgsl += `
const wgsize_x = ${wgsize_x};
const wgsize_y = ${wgsize_y};

@compute @workgroup_size(wgsize_x, wgsize_y)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
  let step = vec2f(1.f / f32(${width_expr}), 1.f / f32(${height_expr}));
`;
    // Emit code to prefetch required data if prefetching is enabled, and load the center pixel.
    if (prefetch === "workgroup") {
        wgsl += `
  // Prefetch the required data to workgroup storage.
  let prefetch_base = vec2i(gid.xy - lid.xy) - ${radius_expr};
  for (var j = i32(lid.y); j < kPrefetchHeight; j += wgsize_y) {
    for (var i = i32(lid.x); i < kPrefetchWidth; i += wgsize_x) {
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
      var norm = 0.f;
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
    // Emit the weight calculations.
    wgsl += `
      norm    = sqrt(f32(i*i) + f32(j*j)) * ${inv_sigma_domain_expr};
      weight  = -0.5f * (norm * norm);

      norm    = distance(pixel.xyz, center_value.xyz) * ${inv_sigma_range_expr};
      weight += -0.5f * (norm * norm);

      weight = exp(weight);
      coeff += weight;
      sum   += weight * pixel;
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
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const center_x = input_data[(x + y * row_stride) * 4 + 0] / 255.0;
            const center_y = input_data[(x + y * row_stride) * 4 + 1] / 255.0;
            const center_z = input_data[(x + y * row_stride) * 4 + 2] / 255.0;
            const center_w = input_data[(x + y * row_stride) * 4 + 3];
            let coeff = 0.0;
            let sum = [0, 0, 0];
            for (let j = -radius; j <= radius; j++) {
                for (let i = -radius; i <= radius; i++) {
                    let xi = Math.min(Math.max(x + i, 0), width - 1);
                    let yj = Math.min(Math.max(y + j, 0), height - 1);
                    let pixel_x = input_data[(xi + yj * row_stride) * 4 + 0] / 255.0;
                    let pixel_y = input_data[(xi + yj * row_stride) * 4 + 1] / 255.0;
                    let pixel_z = input_data[(xi + yj * row_stride) * 4 + 2] / 255.0;
                    let norm;
                    let weight;
                    norm = Math.sqrt(i * i + j * j) / sigma_domain;
                    weight = -0.5 * (norm * norm);
                    let dist_x = pixel_x - center_x;
                    let dist_y = pixel_y - center_y;
                    let dist_z = pixel_z - center_z;
                    norm = Math.sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z) / sigma_range;
                    weight += -0.5 * (norm * norm);
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
// Load the default input image.
LoadInputImage();
export {};
