//! This example shows how to create and work with vector
//! tile layers with style strings containing interpolate and step like expressions

use std::sync::Arc;

use egui::FontDefinitions;
use galileo::control::{EventPropagation, MouseButton, UserEvent, UserEventHandler};
use galileo::layer::data_provider::remove_parameters_modifier;
use galileo::layer::vector_tile_layer::style::VectorTileStyle;
use galileo::layer::vector_tile_layer::VectorTileLayerBuilder;
use galileo::layer::VectorTileLayer;
use galileo::render::text::text_service::TextService;
use galileo::render::text::RustybuzzRasterizer;
use galileo::tile_schema::{TileIndex, TileSchema, TileSchemaBuilder};
use galileo::{Map, MapBuilder};
use galileo_egui::{EguiMap, EguiMapState};
use parking_lot::RwLock;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    run()
}

struct App {
    map: EguiMapState,
    layer: Arc<RwLock<VectorTileLayer>>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            EguiMap::new(&mut self.map).show_ui(ui);
        });

        egui::Window::new("Buttons")
            .title_bar(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Default style").clicked() {
                        self.set_style(default_style());
                    }
                    if ui.button("Interpolated style").clicked() {
                        self.set_style(interpolated_style());
                    }
                });
            });
    }
}

impl App {
    fn new(egui_map_state: EguiMapState, layer: Arc<RwLock<VectorTileLayer>>) -> Self {
        let fonts = FontDefinitions::default();
        let provider = RustybuzzRasterizer::default();

        let text_service = TextService::initialize(provider);
        for font in fonts.font_data.values() {
            text_service.load_font(Arc::new(font.font.to_vec()));
        }

        Self {
            map: egui_map_state,
            layer,
        }
    }

    fn set_style(&mut self, style: VectorTileStyle) {
        let mut layer = self.layer.write();
        if style != *layer.style() {
            layer.update_style(style);
            self.map.request_redraw();
        }
    }
}

pub(crate) fn run() {
    let Some(api_key) = std::option_env!("VT_API_KEY") else {
        panic!("Set the MapTiler API key into VT_API_KEY library when building this example");
    };

    let style = default_style();
    let layer = VectorTileLayerBuilder::new_rest(move |&index: &TileIndex| {
        format!(
            "https://api.maptiler.com/tiles/v3-openmaptiles/{z}/{x}/{y}.pbf?key={api_key}",
            z = index.z,
            x = index.x,
            y = index.y
        )
    })
    .with_style(style)
    .with_tile_schema(tile_schema())
    .with_file_cache_modifier_checked(".tile_cache", Box::new(remove_parameters_modifier))
    .with_attribution(
        "© MapTiler© OpenStreetMap contributors".to_string(),
        "https://www.maptiler.com/copyright/".to_string(),
    )
    .build()
    .expect("failed to create layer");

    let layer = Arc::new(RwLock::new(layer));

    let layer_copy = layer.clone();
    let handler = move |ev: &UserEvent, map: &mut Map| match ev {
        UserEvent::Click(MouseButton::Left, mouse_event) => {
            let view = map.view().clone();
            if let Some(position) = map
                .view()
                .screen_to_map(mouse_event.screen_pointer_position)
            {
                let features = layer_copy.read().get_features_at(&position, &view);

                for (layer, feature) in features {
                    println!("{layer}, {:?}", feature.properties);
                }
            }

            EventPropagation::Stop
        }
        _ => EventPropagation::Propagate,
    };

    let map = MapBuilder::default().with_layer(layer.clone()).build();
    galileo_egui::InitBuilder::new(map)
        .with_handlers([Box::new(handler) as Box<dyn UserEventHandler>])
        .with_app_builder(|egui_map_state, _| Box::new(App::new(egui_map_state, layer)))
        .init()
        .expect("failed to initialize");
}

fn default_style() -> VectorTileStyle {
    serde_json::from_str(include_str!("data/vt_style.json")).expect("invalid style json")
}

fn interpolated_style() -> VectorTileStyle {
    let style_str = r##"
{
  "rules": [
    {
      "symbol": {
        "polygon": {
          "fill_color": [
            "interpolate",
            ["linear"],
            ["zoom"],
            5,  "#e8f4f8ff",
            10, "#b3d9e6ff",
            15, "#7fb3d5ff"
          ]
        }
      }
    },
    {
      "symbol": {
        "line": {
          "stroke_color": [
            "interpolate",
            ["linear"],
            ["zoom"],
            5,  "#d4d4d4ff",
            10, "#8a8a8aff",
            15, "#4a4a4aff"
          ],
          "width": [
            "step",
            ["zoom"],
            0.5,
            8, 1.5,
            12, 3.0,
            16, 5.0
          ]
        }
      }
    },
    {
      "symbol": {
        "point": {
          "color": [
            "interpolate",
            ["linear"],
            ["zoom"],
            6,  "#ff6b6bff",
            8, "#ee5a6fff",
            12, "#c92a2aff"
          ],
          "size": [
            "step",
            ["zoom"],
            4.0,
            10, 10.0,
            14, 12.0,
            18, 14.0
          ]
        }
      }
    }
  ],
  "background": "#f5f5f5ff"
}
"##;
    serde_json::from_str(style_str).expect("invalid style json")
}

fn tile_schema() -> TileSchema {
    TileSchemaBuilder::web_mercator(2..16)
        .with_rect_tile_size(1024)
        .build()
        .expect("invalid tile schema")
}
