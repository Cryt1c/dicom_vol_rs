use dicom_object::open_file;
use dicom_pixeldata::PixelDecoder;
use ndarray::{ArrayBase, Axis, Dim, OwnedRepr};
use rayon::prelude::*;
use three_d::*;

#[tokio::main]
async fn main() {
    run().await;
}

async fn run() {
    let window = Window::new(WindowSettings {
        title: "Hello, world!".to_string(),
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();

    let mut camera = Camera::new_perspective(
        window.viewport(),
        vec3(0.25, -0.5, -2.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        degrees(45.0),
        0.1,
        1000.0,
    );

    let mut control = OrbitControl::new(*camera.target(), 0.25, 100.0);

    let files = std::fs::read_dir("examples/assets/DCM_0000").unwrap();
    let mut sorted_files: Vec<_> = files.filter_map(Result::ok).collect();
    sorted_files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    let mut file_count = 0;

    let loaded_files: Vec<_> = measure_time(
        || {
            sorted_files
                .iter()
                .map(|file| {
                    let file_name = file.file_name();
                    let file_name = file_name.to_str().unwrap();
                    let file =
                        open_file(&format!("examples/assets/DCM_0000/{}", file_name)).unwrap();
                    file_count += 1;
                    return file;
                })
                .collect()
        },
        "File load time",
    );

    let decoded_pixel_data = measure_time(
        || {
            loaded_files
                .par_iter()
                .map(|file| {
                    let pixel_data = file.decode_pixel_data().unwrap();
                    return pixel_data;
                })
                .collect::<Vec<_>>()
        },
        "Decode pixel time",
    );

    let arrays: Vec<ArrayBase<OwnedRepr<f16>, Dim<[usize; 4]>>> = measure_time(
        || {
            decoded_pixel_data
                .into_par_iter()
                .map(|data| data.to_ndarray::<f16>().unwrap())
                .collect()
        },
        "Convert to ndarray time",
    );

    let array_views: Vec<_> = measure_time(
        || arrays.iter().map(|array| array.view()).collect(),
        "Create views time",
    );

    let concat_array = measure_time(
        || ndarray::stack(Axis(2), &array_views).unwrap(),
        "Concat time",
    );

    let now = std::time::Instant::now();
    let cpu_voxel_grid = CpuVoxelGrid {
        voxels: CpuTexture3D {
            name: "sample".to_string(),
            data: TextureData::RF16(concat_array.into_raw_vec()),
            width: 512,
            height: 512,
            depth: file_count,
            min_filter: Interpolation::Linear,
            mag_filter: Interpolation::Linear,
            wrap_s: Wrapping::Repeat,
            wrap_t: Wrapping::Repeat,
            wrap_r: Wrapping::Repeat,
            mip_map_filter: None,
        },
        ..Default::default()
    };
    let mut voxel_grid = VoxelGrid::<IsosurfaceMaterial>::new(&context, &cpu_voxel_grid);

    let ambient = AmbientLight::new(&context, 0.4, Srgba::WHITE);
    let directional1 = DirectionalLight::new(&context, 2.0, Srgba::WHITE, &vec3(-1.0, -1.0, -1.0));
    let directional2 = DirectionalLight::new(&context, 2.0, Srgba::WHITE, &vec3(1.0, 1.0, 1.0));

    let mut gui = three_d::GUI::new(&context);
    let mut color = [1.0; 4];
    println!("Render prep time: {:?}", now.elapsed());

    window.render_loop(move |mut frame_input| {
        let mut panel_width = 0.0;
        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |gui_context| {
                use three_d::egui::*;
                SidePanel::left("side_panel").show(gui_context, |ui| {
                    ui.heading("Debug Panel");
                    ui.add(
                        Slider::new(&mut voxel_grid.material.threshold, -1000.0..=3000.0)
                            .text("Threshold"),
                    );
                    ui.color_edit_button_rgba_unmultiplied(&mut color);
                });
                panel_width = gui_context.used_rect().width();
            },
        );
        voxel_grid.material.color = Srgba::from(color);

        let viewport = Viewport {
            x: (panel_width * frame_input.device_pixel_ratio) as i32,
            y: 0,
            width: frame_input.viewport.width
                - (panel_width * frame_input.device_pixel_ratio) as u32,
            height: frame_input.viewport.height,
        };
        camera.set_viewport(viewport);

        control.handle_events(&mut camera, &mut frame_input.events);

        frame_input
            .screen()
            .clear(ClearState::color_and_depth(0.5, 0.5, 0.5, 1.0, 1.0))
            .render(
                &camera,
                &voxel_grid,
                &[&ambient, &directional1, &directional2],
            )
            .write(|| gui.render());

        FrameOutput::default()
    });
}

fn measure_time<T, F: FnOnce() -> T>(f: F, name: &str) -> T {
    let now = std::time::Instant::now();
    let result = f();
    println!("{}: {:?}", name, now.elapsed());
    return result;
}
