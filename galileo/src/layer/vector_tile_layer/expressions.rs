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
#[derive(Clone, Debug, Serialize, Deserialize, PartialOrd, PartialEq)]
pub struct StepExpression<T: Ord> {
    default_value: T,
    /// Each stop value maps the resolution to the T type
    /// If, the current resolution is greater than step resolution
    /// the value T maps to the T value where the step resolution is
    /// less than that of current resolution.
    step_resolution: Vec<f64>,
    step_value_: Vec<T>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialOrd, PartialEq)]
pub struct ValueStep<T> {
    resolution: f64,
    value: T,
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
pub enum StyleValue<T: Ord> {
    /// Simple values like using Color::BLACK
    Simple(T),
    /// Interpolate expression
    Interpolate(InterpolateExpression<T>),
    /// Step expression
    Steps(StepExpression<T>),
}

impl Color {
    /// Evaluates value from simple color
    pub fn get_value(&self, _: InterpolateContext) -> Result<Color, GalileoError> {
        Ok(*self)
    }
}
impl InterpolateExpression<Color> {
    /// Evaluates value by interpolating color values
    pub fn get_value(&self, current_resolution: f64) -> Result<Color, GalileoError> {
        match self.interpolation_type {
            Interpolation::Linear => {
                let interpolated_color: Color = self.lerp(current_resolution)?;
                Ok(interpolated_color)
            }
            Interpolation::Exponential => {
                let interpolated_color: Color = self.exponential_interpolate(current_resolution)?;
                Ok(interpolated_color)
            }
            Interpolation::Cubic => {
                let interpolated_color: Color = self.cubic_interpolate(current_resolution)?;
                todo!()
            }
        }
    }
    fn lerp(&self, current_resolution: f64) -> Result<Color, GalileoError> {
        match (self.max_resolution, self.min_resolution) {
            (Some(max_resolution), Some(min_resolution)) => {
                let resolution_range: f64 =
                    (max_resolution - min_resolution).clamp(f64::EPSILON, f64::MAX);
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
    fn exponential_interpolate(&self, current_resolution: f64) -> Result<Color, GalileoError> {
        match (self.max_resolution, self.min_resolution) {
            (Some(max_resolution), Some(min_resolution)) => {
                if let Some(interpolation_args) = &self.interpolation_args {
                    if interpolation_args.len() != 1 {
                        Err(GalileoError::Configuration(
                            "Ill populated interpolation arguments".to_string(),
                        ))
                    } else {
                        let t: f64 = ((current_resolution - min_resolution)
                            / (max_resolution - min_resolution).clamp(f64::EPSILON, f64::MAX))
                        .clamp(0.0, 1.0);
                        let base: f64 = interpolation_args[0] as f64;

                        let t = if (base - 1.0).abs() > f64::EPSILON {
                            (base.powf(t) - 1.0) / (base - 1.0)
                        } else {
                            t
                        };
                        let offset_r = (self.end_value.r() - self.start_value.r()) as f64;
                        let offset_g = (self.end_value.g() - self.start_value.g()) as f64;
                        let offset_b = (self.end_value.b() - self.start_value.b()) as f64;
                        let offset_a = (self.end_value.a() - self.start_value.a()) as f64;
                        Ok(Color::rgba(
                            (self.start_value.r() as f64 + t * (offset_r)).clamp(0.0, 255.0) as u8,
                            (self.start_value.g() as f64 + t * (offset_g)).clamp(0.0, 255.0) as u8,
                            (self.start_value.b() as f64 + t * (offset_b)).clamp(0.0, 255.0) as u8,
                            (self.start_value.a() as f64 + t * (offset_a)).clamp(0.0, 255.0) as u8,
                        ))
                    }
                } else {
                    Err(GalileoError::Configuration(
                        "Missing resolution configurations!".to_string(),
                    ))
                }
            }
            (_, _) => Err(GalileoError::Configuration(
                "Missing resolution configurations!".to_string(),
            ))?,
        }
    }
}
impl StepExpression<Color> {
    /// Evaluates color value by giving stepwise value
    /// of color on basis of zoom
    pub fn get_value(&self, _resolution: f64) -> Result<Color, GalileoError> {
        Ok(Color::BLACK)
    }
}
