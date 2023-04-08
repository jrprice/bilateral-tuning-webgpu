import { ConfigRunner } from "./config_runner.js";
// The input image data.
let width;
let height;
// WebGPU objects.
let adapter;
let device;
let input_image_staging;
// The shader config runner object.
let runner;
function SetStatus(str, color = "#000000") {
    document.getElementById("status").textContent = str;
    document.getElementById("status").style.color = color;
}
function SetRuntime(str) {
    document.getElementById("runtime").textContent = str;
}
/// Initialize the main WebGPU objects.
async function InitWebGPU() {
    SetStatus("Initializing...");
    // Check for WebGPU support.
    if (!navigator.gpu) {
        SetStatus("WebGPU is not supported in this browser!", "#FF0000");
        document.getElementById("run").disabled = true;
        throw "WebGPU is not supported in this browser.";
    }
    // Initialize the WebGPU adapter and device.
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
    // Prepare the config runner for the new input image.
    await runner.SetupTextures(input_bitmap);
    // Update the canvases.
    DisplayTexture(runner.input_image_staging, "input_canvas");
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
    SetStatus("Ready.");
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
/// Get the shader config from the HTML input elements.
function GetShaderConfigFromForm() {
    let config = {
        wgsize_x: +document.getElementById("wgsize_x").value,
        wgsize_y: +document.getElementById("wgsize_y").value,
        tilesize_x: +document.getElementById("tilesize_x").value,
        tilesize_y: +document.getElementById("tilesize_y").value,
        const_sigma_domain: document.getElementById("const_sd").checked,
        const_sigma_range: document.getElementById("const_sr").checked,
        const_radius: document.getElementById("const_radius").checked,
        const_width: document.getElementById("const_width").checked,
        const_height: document.getElementById("const_height").checked,
        input_type: document.getElementById("input_type").value,
        prefetch: document.getElementById("prefetch").value,
        spatial_coeffs: document.getElementById("spatial_coeffs").value,
    };
    return config;
}
/// Run the benchmark.
const Run = async () => {
    SetRuntime("");
    await runner.RunConfig({ config: GetShaderConfigFromForm() });
    DisplayTexture(runner.output_texture, "output_canvas");
    DisplayImageData(runner.reference_data, "reference_canvas");
    DisplayImageData(runner.diff_data, "diff_canvas");
};
/// Test all configs to check for issues.
const Test = async () => {
    SetRuntime("");
    // Helper to change a radio button selection.
    function ConstUniformRadio(name, uniform) {
        if (uniform) {
            document.getElementById(`uniform_${name}`).click();
        }
        else {
            document.getElementById(`const_${name}`).click();
        }
    }
    // Helper to change a drop-down selection.
    function SelectDropDown(name, value) {
        const select = document.getElementById(name);
        select.value = value;
        select.dispatchEvent(new Event("change"));
    }
    const start = performance.now();
    let num_configs = 0;
    // TODO: test non-square workgroup sizes
    for (const tile_width of ["1", "2"]) {
        SelectDropDown("tilesize_x", tile_width);
        for (const tile_height of ["1", "2"]) {
            SelectDropDown("tilesize_y", tile_height);
            for (const uniform_sigma_domain of [true, false]) {
                ConstUniformRadio("sd", uniform_sigma_domain);
                for (const uniform_sigma_range of [true, false]) {
                    ConstUniformRadio("sr", uniform_sigma_range);
                    for (const uniform_radius of [true, false]) {
                        ConstUniformRadio("radius", uniform_radius);
                        for (const uniform_width of [true, false]) {
                            ConstUniformRadio("width", uniform_width);
                            for (const uniform_height of [true, false]) {
                                ConstUniformRadio("height", uniform_height);
                                for (const input_type of ["image_sample", "image_load"]) {
                                    SelectDropDown("input_type", input_type);
                                    for (const prefetch of ["none", "workgroup"]) {
                                        SelectDropDown("prefetch", prefetch);
                                        for (const spatial_coeffs of ["inline", "lut_uniform", "lut_const"]) {
                                            SelectDropDown("spatial_coeffs", spatial_coeffs);
                                            // Skip invalid configs.
                                            if (uniform_radius && prefetch !== "none") {
                                                continue;
                                            }
                                            // Run the config and check the result.
                                            if (!(await runner.RunConfig({ config: GetShaderConfigFromForm(), test: true }))) {
                                                SetStatus("Config failed!", "#FF0000");
                                                return;
                                            }
                                            num_configs++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    const end = performance.now();
    const elapsed = end - start;
    SetRuntime(`Tested ${num_configs} configurations in ${(elapsed / 1000).toFixed(1)} seconds`);
};
/// Update and display the WGSL shader.
function UpdateShader(config) {
    const shader_display = document.getElementById("shader");
    shader_display.style.width = `0px`;
    shader_display.style.height = `0px`;
    shader_display.textContent = runner.GenerateShader(config);
    shader_display.style.width = `${shader_display.scrollWidth}px`;
    shader_display.style.height = `${shader_display.scrollHeight}px`;
}
// Initialize WebGPU.
await InitWebGPU();
// Create the shader config runner.
runner = new ConfigRunner(device);
runner.UpdateStatusCallback = SetStatus;
runner.UpdateRuntimeCallback = SetRuntime;
// Load the default input image.
await LoadInputImage();
// Display the default shader.
UpdateShader(GetShaderConfigFromForm());
// Add event handlers for the buttons.
document.querySelector("#run").addEventListener("click", Run);
document.querySelector("#test").addEventListener("click", Test);
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
    runner.SetRadius(+document.getElementById("radius").value);
    UpdateShader(GetShaderConfigFromForm());
});
// Add event handlers for the shader parameter radio buttons.
document.querySelector("#const_sd").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
document.querySelector("#uniform_sd").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
document.querySelector("#const_sr").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
document.querySelector("#uniform_sr").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
document.querySelector("#const_radius").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
document.querySelector("#uniform_radius").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
document.querySelector("#const_width").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
document.querySelector("#uniform_width").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
document.querySelector("#const_height").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
document.querySelector("#uniform_height").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
// Add event handlers for the shader parameter drop-down menus.
document.querySelector("#wgsize_x").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
document.querySelector("#wgsize_y").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
document.querySelector("#tilesize_x").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
document.querySelector("#tilesize_y").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
document.querySelector("#input_type").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
document.querySelector("#prefetch").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
document.querySelector("#spatial_coeffs").addEventListener("change", () => {
    UpdateShader(GetShaderConfigFromForm());
});
