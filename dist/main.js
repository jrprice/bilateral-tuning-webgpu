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
    if (!adapter) {
        SetStatus("Failed to create WebGPU adapter!", "#FF0000");
        document.getElementById("run").disabled = true;
        throw "Failed to create WebGPU adapter.";
    }
    device = await adapter.requestDevice();
    // Create the shader config runner.
    runner = new ConfigRunner(device);
    runner.UpdateStatusCallback = SetStatus;
    runner.UpdateRuntimeCallback = SetRuntime;
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
/// Push the shader config to the HTML input elements.
function PushShaderConfigToForm(config) {
    // Helper to change a radio button selection.
    function ConstUniformRadio(name, uniform) {
        if (uniform) {
            document.getElementById(`uniform_${name}`).checked = true;
        }
        else {
            document.getElementById(`const_${name}`).checked = true;
        }
    }
    // Helper to change a drop-down selection.
    function SelectDropDown(name, value) {
        document.getElementById(name).value = value;
    }
    SelectDropDown("wgsize_x", config.wgsize_x.toString());
    SelectDropDown("wgsize_y", config.wgsize_y.toString());
    SelectDropDown("tilesize_x", config.tilesize_x.toString());
    SelectDropDown("tilesize_y", config.tilesize_y.toString());
    ConstUniformRadio("sd", !config.const_sigma_domain);
    ConstUniformRadio("sr", !config.const_sigma_range);
    ConstUniformRadio("radius", !config.const_radius);
    ConstUniformRadio("width", !config.const_width);
    ConstUniformRadio("height", !config.const_height);
    SelectDropDown("input_type", config.input_type);
    SelectDropDown("prefetch", config.prefetch);
    SelectDropDown("spatial_coeffs", config.spatial_coeffs);
}
/// Run the benchmark.
const Run = async () => {
    SetRuntime("");
    const iterations = +document.getElementById("iterations").value;
    await runner.RunConfig({ config: GetShaderConfigFromForm(), iterations });
    DisplayTexture(runner.output_texture, "output_canvas");
    DisplayImageData(runner.reference_data, "reference_canvas");
    DisplayImageData(runner.diff_data, "diff_canvas");
};
/// Test all configs to check for issues.
const Test = async () => {
    SetRuntime("");
    const start = performance.now();
    let num_configs = 0;
    let config = {};
    // TODO: test non-square workgroup sizes
    config.wgsize_x = 8;
    config.wgsize_y = 8;
    for (const tile_width of ["1", "2"]) {
        config.tilesize_x = +tile_width;
        for (const tile_height of ["1", "2"]) {
            config.tilesize_y = +tile_height;
            for (const uniform_sigma_domain of [true, false]) {
                config.const_sigma_domain = !uniform_sigma_domain;
                for (const uniform_sigma_range of [true, false]) {
                    config.const_sigma_range = !uniform_sigma_range;
                    for (const uniform_radius of [true, false]) {
                        config.const_radius = !uniform_radius;
                        for (const uniform_width of [true, false]) {
                            config.const_width = !uniform_width;
                            for (const uniform_height of [true, false]) {
                                config.const_height = !uniform_height;
                                for (const input_type of ["image_sample", "image_load"]) {
                                    config.input_type = input_type;
                                    for (const prefetch of ["none", "workgroup"]) {
                                        config.prefetch = prefetch;
                                        for (const spatial_coeffs of ["inline", "lut_uniform", "lut_const"]) {
                                            config.spatial_coeffs = spatial_coeffs;
                                            // Skip invalid configs.
                                            if (!config.const_radius && config.prefetch !== "none") {
                                                continue;
                                            }
                                            PushShaderConfigToForm(config);
                                            UpdateShader(config);
                                            // Run the config and check the result.
                                            if (!(await runner.RunConfig({
                                                config,
                                                iterations: 1,
                                                test: true,
                                            })).validated) {
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
/// Generate the immediate neighbors of a shader config.
function GenerateNeighbors(config) {
    let neighbors = [];
    // Push a config to the neighbor list, if valid.
    function Push(new_config) {
        // Skip invalid configs.
        if (!new_config.const_radius && new_config.prefetch !== "none") {
            return;
        }
        if (new_config.wgsize_x * new_config.wgsize_y < 8) {
            return;
        }
        neighbors.push(new_config);
    }
    // Mutate a boolean parameter.
    function MutateBool(param) {
        let result = Object.assign({}, config);
        result[param] = !result[param];
        Push(result);
    }
    // Mutate an enum parameter.
    function MutateEnum(param, values) {
        for (const v of values) {
            if (config[param] === v) {
                continue;
            }
            let result = Object.assign({}, config);
            result[param] = v;
            Push(result);
        }
    }
    // Mutate a power-of-2 parameter.
    function MutatePow2(param, max) {
        if (config[param] > 1) {
            let result = Object.assign({}, config);
            result[param] /= 2;
            Push(result);
        }
        if (config[param] < max) {
            let result = Object.assign({}, config);
            result[param] *= 2;
            Push(result);
        }
    }
    MutateBool("const_sigma_domain");
    MutateBool("const_sigma_range");
    MutateBool("const_radius");
    MutateBool("const_width");
    MutateBool("const_height");
    MutateEnum("input_type", ["image_sample", "image_load"]);
    MutateEnum("prefetch", ["none", "workgroup"]);
    MutateEnum("spatial_coeffs", ["inline", "lut_uniform", "lut_const"]);
    MutatePow2("tilesize_x", 16);
    MutatePow2("tilesize_y", 16);
    MutatePow2("wgsize_x", 256);
    MutatePow2("wgsize_y", 256);
    return neighbors;
}
/// Run the tuning process.
const Tune = async () => {
    const old_status_callback = runner.UpdateStatusCallback;
    const old_runtime_callback = runner.UpdateRuntimeCallback;
    runner.UpdateStatusCallback = () => { };
    runner.UpdateRuntimeCallback = () => { };
    SetStatus("");
    SetRuntime("");
    const iterations = +document.getElementById("tune_iterations").value;
    // Run a config and return the achieved FPS.
    async function Run(config) {
        const result = await runner.RunConfig({ config, iterations, test: false });
        if (!result.validated) {
            console.log(`Config failed: ${config}`);
            return null;
        }
        return result.fps;
    }
    let current_config = GetShaderConfigFromForm();
    let current_fps = await Run(current_config);
    SetRuntime(`Initial FPS = ${current_fps.toFixed(1)}`);
    let rounds = 1;
    while (true) {
        let new_best_config = null;
        let neighbors = GenerateNeighbors(current_config);
        for (var c = 0; c < neighbors.length; c++) {
            const n = neighbors[c];
            SetStatus(`Round ${rounds} (config ${c + 1}/${neighbors.length})`);
            PushShaderConfigToForm(n);
            UpdateShader(n);
            // Run the config.
            const fps = await Run(n);
            if (fps > current_fps) {
                new_best_config = Object.assign({}, n);
                current_fps = fps;
                SetRuntime(`Current best: ${current_fps.toFixed(1)} FPS`);
            }
        }
        if (!new_best_config) {
            break;
        }
        current_config = Object.assign({}, new_best_config);
        rounds++;
    }
    SetStatus(`Tuning finished after ${rounds} rounds`);
    SetRuntime(`Best performance: ${current_fps.toFixed(1)} FPS`);
    PushShaderConfigToForm(current_config);
    runner.UpdateStatusCallback = old_status_callback;
    runner.UpdateRuntimeCallback = old_runtime_callback;
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
// Load the default input image.
await LoadInputImage();
// Display the default shader.
UpdateShader(GetShaderConfigFromForm());
// Add event handlers for the buttons.
document.querySelector("#run").addEventListener("click", Run);
document.querySelector("#test").addEventListener("click", Test);
document.querySelector("#tune").addEventListener("click", Tune);
// Add an event handler for the power preference selector.
document.querySelector("#powerpref").addEventListener("change", () => {
    InitWebGPU().then(() => {
        LoadInputImage();
    });
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
