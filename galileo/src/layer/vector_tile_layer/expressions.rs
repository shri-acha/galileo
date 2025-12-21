use core::f64;

use serde::{Deserialize, Serialize};

use crate::error::GalileoError;
use crate::Color;

/// Context for Step
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StepContext {
    pub current_resolution: f64,
}

/// Context for the interpolation
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InterpolateContext {
    pub current_resolution: f64,
}

/// Type used to define expressions for interpolation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InterpolateExpression<T> {
    start_value: T,
    end_value: T,
    max_resolution: Option<f64>,
    min_resolution: Option<f64>,
    interpolation_type: Interpolation,
    interpolation_args: Option<Vec<i32>>,
}
/// Type used to define steps
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StepExpression<T> {
    default_value: T,
    /// Each stop value maps the resolution to the T type
    /// If, the current resolution is greater than stop resolution
    /// the value T maps to the T value where the stop resolution is
    /// less than that of current resolution.
    stop_values_resolution: Vec<f64>,
    stop_values_type: Vec<T>,
}
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Interpolation {
    /// Linear interpolation type with base 1
    Linear,
    /// Exponential interpolation with variable base
    Exponential,
    /// Cubic Bezier interpolation
    Cubic,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub enum StyleValue<T> {
    Simple(T),
    Interpolate(InterpolateExpression<T>),
    Steps(StepExpression<T>),
}

impl Color {
    pub fn get_value(&self, _: InterpolateContext) -> Result<Color, GalileoError> {
        Ok(*self)
    }
}

impl InterpolateExpression<Color> {
    pub fn get_value(&self, context: InterpolateContext) -> Result<Color, GalileoError> {
        match self.interpolation_type {
            Interpolation::Linear => {
                let interpolated_color: Color = self.lerp(context.current_resolution)?;
                Ok(interpolated_color)
            }
            Interpolation::Exponential => {
                todo!()
            }
            Interpolation::Cubic => {
                todo!()
            }
        }
    }
    fn lerp(&self, current_resolution: f64) -> Result<Color, GalileoError> {
        match (self.max_resolution, self.min_resolution) {
            (Some(max_resolution), Some(min_resolution)) => {
                const EPS: f64 = 10e-6;

                let resolution_range: f64 = (max_resolution - min_resolution).clamp(EPS, f64::MAX);
                // individual ratios for each field
                let kr = (self.end_value.r() - self.start_value.r()) as f64 / resolution_range;
                let kg = (self.end_value.g() - self.start_value.g()) as f64 / resolution_range;
                let kb = (self.end_value.b() - self.start_value.b()) as f64 / resolution_range;
                let ka = (self.end_value.a() - self.start_value.a()) as f64 / resolution_range;

                let offset = (current_resolution - min_resolution).clamp(0.0, resolution_range);

                Ok(Color::rgba(
                    (self.start_value.r() as f64 + kr * offset).clamp(0.0, 255.0) as u8,
                    (self.start_value.g() as f64 + kg * offset).clamp(0.0, 255.0) as u8,
                    (self.start_value.b() as f64 + kb * offset).clamp(0.0, 255.0) as u8,
                    (self.start_value.a() as f64 + ka * offset).clamp(0.0, 255.0) as u8,
                ))
            }
            (_, _) => Err(GalileoError::Configuration(
                "Unexpectedly missing resolution configurations!".to_string(),
            ))?,
        }
    }
}
impl StepExpression<Color> {
    pub fn get_value(&self, _context: StepContext) -> Result<Color, GalileoError> {
        Ok(Color::BLACK)
    }
}
