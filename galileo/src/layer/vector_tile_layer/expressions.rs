use serde::{Deserialize, Serialize};

use crate::error::GalileoError;
use crate::Color;

/// Arguments for exponential interpolation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearArgs<T> {
    step_values: Option<Vec<StepValue<T>>>,
}

/// Arguments for exponential interpolation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepArgs<T> {
    step_values: Option<Vec<StepValue<T>>>,
}

/// Generic arguments for interpolation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InterpolationArgs<T> {
    base: Option<i32>,
    /// Step values is internally sorted
    step_values: Option<Vec<StepValue<T>>>,
}
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StepValue<T> {
    resolution: f64,
    step_value: T,
}

/// Type used to define expressions for interpolation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InterpolateExpression<T> {
    start_value: T,
    end_value: T,
    max_resolution: Option<f64>,
    min_resolution: Option<f64>,
    interpolation_type: Interpolation,
    interpolation_args: InterpolationArgs<T>,
}
/// Type used to define steps
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StepExpression<T> {
    default_value: T,
    /// Each stop value maps the resolution to the T type
    /// If, the current resolution is greater than step resolution
    /// the value T maps to the T value where the step resolution is
    /// less than that of current resolution.
    step_values: Option<Vec<StepValue<T>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ResolutionValueRange<T> {
    max_resolution: f64,
    min_resolution: f64,
    start_value: T,
    end_value: T,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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
    pub fn get_value(&self) -> Result<Color, GalileoError> {
        Ok(*self)
    }
}

impl InterpolateExpression<Color> {
    /// Evaluates value by interpolating color values
    pub fn get_value(&self, current_resolution: f64) -> Option<Color> {
        let resolution_value_range: Option<ResolutionValueRange<Color>> =
            self.get_resolution_value_range(current_resolution);

        if let Some(resolution_value_range) = resolution_value_range {
            match self.interpolation_type {
                Interpolation::Linear => {
                    let interpolated_color: Color = Self::lerp(
                        resolution_value_range.start_value,
                        resolution_value_range.end_value,
                        Some(resolution_value_range.min_resolution),
                        Some(resolution_value_range.max_resolution),
                        current_resolution,
                    )?;
                    Some(interpolated_color)
                }
                Interpolation::Exponential => {
                    let interpolated_color: Color = Self::exponential_interpolate(
                        resolution_value_range.start_value,
                        resolution_value_range.end_value,
                        Some(resolution_value_range.min_resolution),
                        Some(resolution_value_range.max_resolution),
                        &self.interpolation_args,
                        current_resolution,
                    )?;
                    Some(interpolated_color)
                }
                Interpolation::Cubic => {
                    todo!()
                }
            }
        } else {
            None
        }
    }

    fn lerp(
        start_value: Color,
        end_value: Color,
        min_resolution: Option<f64>,
        max_resolution: Option<f64>,
        current_resolution: f64,
    ) -> Option<Color> {
        match (max_resolution, min_resolution) {
            (Some(max_resolution), Some(min_resolution)) => {
                let resolution_range: f64 =
                    (max_resolution - min_resolution).clamp(f64::EPSILON, f64::MAX);
                // individual ratios for each field
                let kr = (end_value.r() - start_value.r()) as f64 / resolution_range;
                let kg = (end_value.g() - start_value.g()) as f64 / resolution_range;
                let kb = (end_value.b() - start_value.b()) as f64 / resolution_range;
                let ka = (end_value.a() - start_value.a()) as f64 / resolution_range;

                let offset = (current_resolution - min_resolution).clamp(0.0, resolution_range);

                Some(Color::rgba(
                    (start_value.r() as f64 + kr * offset).clamp(0.0, 255.0) as u8,
                    (start_value.g() as f64 + kg * offset).clamp(0.0, 255.0) as u8,
                    (start_value.b() as f64 + kb * offset).clamp(0.0, 255.0) as u8,
                    (start_value.a() as f64 + ka * offset).clamp(0.0, 255.0) as u8,
                ))
            }
            (_, _) => None,
        }
    }

    fn exponential_interpolate(
        start_value: Color,
        end_value: Color,
        min_resolution: Option<f64>,
        max_resolution: Option<f64>,
        interpolation_args: &InterpolationArgs<Color>,
        current_resolution: f64,
    ) -> Option<Color> {
        match (max_resolution, min_resolution) {
            (Some(max_resolution), Some(min_resolution)) => {
                let t: f64 = ((current_resolution - min_resolution)
                    / (max_resolution - min_resolution).clamp(f64::EPSILON, f64::MAX))
                .clamp(0.0, 1.0);

                if let Some(base) = interpolation_args.base {
                    let base: f64 = base as f64;
                    let t = if (base - 1.0).abs() > f64::EPSILON {
                        (base.powf(t) - 1.0) / (base - 1.0)
                    } else {
                        t
                    };

                    let offset_r = (end_value.r() - start_value.r()) as f64;
                    let offset_g = (end_value.g() - start_value.g()) as f64;
                    let offset_b = (end_value.b() - start_value.b()) as f64;
                    let offset_a = (end_value.a() - start_value.a()) as f64;

                    Some(Color::rgba(
                        (start_value.r() as f64 + t * (offset_r)).clamp(0.0, 255.0) as u8,
                        (start_value.g() as f64 + t * (offset_g)).clamp(0.0, 255.0) as u8,
                        (start_value.b() as f64 + t * (offset_b)).clamp(0.0, 255.0) as u8,
                        (start_value.a() as f64 + t * (offset_a)).clamp(0.0, 255.0) as u8,
                    ))
                } else {
                    None
                }
            }
            (_, _) => None,
        }
    }

    fn cubic_interpolate(
        step_values: Vec<&StepValue<Color>>,
        min_resolution: Option<f64>,
        max_resolution: Option<f64>,
        interpolation_args: &InterpolationArgs<Color>,
        current_resolution: f64,
    ) -> Option<Color> {
        todo!();
    }

    fn get_resolution_value_range(
        &self,
        current_resolution: f64,
    ) -> Option<ResolutionValueRange<Color>> {
        // checks if within the min/max bounds
        if let (Some(min_res), Some(max_res)) = (self.min_resolution, self.max_resolution) {
            if current_resolution < min_res || current_resolution > max_res {
                return None;
            }
        }

        let step_values = self.interpolation_args.step_values.as_ref()?;

        // Try to find a matching window in step_values
        if let Some(window) = step_values.windows(2).find(|w| {
            current_resolution >= w[0].resolution && current_resolution <= w[1].resolution
        }) {
            return Some(ResolutionValueRange {
                min_resolution: window[0].resolution,
                max_resolution: window[1].resolution,
                start_value: window[0].step_value,
                end_value: window[1].step_value,
            });
        }

        // If not found in step_values, check if before the first step
        if let Some(first_step) = step_values.first() {
            if let Some(min_res) = self.min_resolution {
                if current_resolution >= min_res && current_resolution <= first_step.resolution {
                    return Some(ResolutionValueRange {
                        min_resolution: min_res,
                        max_resolution: first_step.resolution,
                        start_value: self.start_value,
                        end_value: first_step.step_value,
                    });
                }
            }
        }

        // If not found in step_values, check if after last step
        if let Some(last_step) = step_values.last() {
            if let Some(max_res) = self.max_resolution {
                if current_resolution >= last_step.resolution && current_resolution <= max_res {
                    return Some(ResolutionValueRange {
                        min_resolution: last_step.resolution,
                        max_resolution: max_res,
                        start_value: last_step.step_value,
                        end_value: self.end_value,
                    });
                }
            }
        }

        None
    }
}

impl StepExpression<Color> {
    /// Evaluates color value by giving stepwise value
    /// of color on basis of zoom
    pub fn get_value(&self, current_resolution: f64) -> Option<Color> {
        if let Some(step_values) = self.step_values.as_ref() {
            if step_values.len() < 1 {
                return None;
            }
            // Value when
            if let Some(w) = step_values.windows(2).find(|w| {
                current_resolution >= w[0].resolution && current_resolution <= w[1].resolution
            }) {
                return Some(w[0].step_value);
            } else if current_resolution < step_values[0].resolution {
                return Some(self.default_value);
            } else {
                return Some(step_values[step_values.len() - 1].step_value);
            }
        } else {
            return None;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_get_resolution_value_range_out_of_bounds() {
        let expr = InterpolateExpression {
            start_value: Color::rgba(0, 0, 0, 0),
            end_value: Color::rgba(255, 255, 255, 255),
            min_resolution: Some(0.0),
            max_resolution: Some(100.0),
            interpolation_type: Interpolation::Linear,
            interpolation_args: InterpolationArgs {
                base: None,
                step_values: Some(vec![
                    StepValue {
                        resolution: 0.0,
                        step_value: Color::rgba(0, 0, 0, 0),
                    },
                    StepValue {
                        resolution: 50.0,
                        step_value: Color::rgba(128, 128, 128, 128),
                    },
                ]),
            },
        };

        let range = expr.get_resolution_value_range(75.0);
        assert!(range.is_some());
    }

    #[test]
    fn linear_interpolation_bounds() {
        let expr = InterpolateExpression {
            start_value: Color::rgba(0, 0, 0, 0),
            end_value: Color::rgba(255, 255, 255, 255),
            min_resolution: Some(50.0),
            max_resolution: Some(100.0),
            interpolation_type: Interpolation::Linear,
            interpolation_args: InterpolationArgs {
                base: None,
                step_values: Some(vec![
                    StepValue {
                        resolution: 0.0,
                        step_value: Color::rgba(0, 0, 0, 0),
                    },
                    StepValue {
                        resolution: 50.0,
                        step_value: Color::rgba(128, 128, 128, 128),
                    },
                ]),
            },
        };
        assert_eq!(expr.get_value(25.0), None);
        assert_eq!(expr.get_value(150.0), None);
    }
    #[test]
    fn linear_interpolation() {
        let expr = InterpolateExpression {
            start_value: Color::rgba(0, 0, 0, 0),
            end_value: Color::rgba(255, 255, 255, 255),
            min_resolution: Some(0.0),
            max_resolution: Some(100.0),
            interpolation_type: Interpolation::Linear,
            interpolation_args: InterpolationArgs {
                base: None,
                step_values: Some(vec![
                    StepValue {
                        resolution: 0.0,
                        step_value: Color::rgba(0, 0, 0, 0),
                    },
                    StepValue {
                        resolution: 50.0,
                        step_value: Color::rgba(128, 128, 128, 128),
                    },
                ]),
            },
        };
        assert_eq!(expr.get_value(25.0), Some(Color::rgba(64, 64, 64, 64)));
    }
    #[test]
    fn test_exponential_bounds() {
        let expr = InterpolateExpression {
            start_value: Color::rgba(0, 0, 0, 0),
            end_value: Color::rgba(255, 255, 255, 255),
            min_resolution: Some(50.0),
            max_resolution: Some(100.0),
            interpolation_type: Interpolation::Exponential,
            interpolation_args: InterpolationArgs {
                base: Some(2),
                step_values: Some(vec![
                    StepValue {
                        resolution: 0.0,
                        step_value: Color::rgba(0, 0, 0, 0),
                    },
                    StepValue {
                        resolution: 50.0,
                        step_value: Color::rgba(128, 128, 128, 128),
                    },
                ]),
            },
        };
        assert_eq!(expr.get_value(5.0), None);
        assert_eq!(expr.get_value(150.0), None);
    }

    #[test]
    fn exponential_interpolation() {
        let expr = InterpolateExpression {
            start_value: Color::rgba(0, 0, 0, 0),
            end_value: Color::rgba(255, 255, 255, 255),
            min_resolution: Some(10.0),
            max_resolution: Some(100.0),
            interpolation_type: Interpolation::Exponential,
            interpolation_args: InterpolationArgs {
                base: Some(2),
                step_values: Some(vec![
                    StepValue {
                        resolution: 0.0,
                        step_value: Color::rgba(0, 0, 0, 0),
                    },
                    StepValue {
                        resolution: 50.0,
                        step_value: Color::rgba(128, 128, 128, 128),
                    },
                    StepValue {
                        resolution: 75.0,
                        step_value: Color::rgba(200, 200, 200, 200),
                    },
                ]),
            },
        };
        assert_eq!(expr.get_value(25.0), Some(Color::rgba(53, 53, 53, 53)));
        assert_eq!(expr.get_value(60.0), Some(Color::rgba(151, 151, 151, 151)));
    }

    #[test]
    fn test_step_expression_bounds() {
        let expr = StepExpression::<Color> {
            default_value: Color::from_hex("#f0f0f0"),
            step_values: Some(vec![
                StepValue::<Color> {
                    resolution: 10.0,
                    step_value: Color::from_hex("#fafafa"),
                },
                StepValue::<Color> {
                    resolution: 20.0,
                    step_value: Color::from_hex("#1d1d1d"),
                },
            ]),
        };
        assert_eq!(expr.get_value(0.0), Some(Color::from_hex("#f0f0f0")));
        assert_eq!(expr.get_value(30.0), Some(Color::from_hex("#1d1d1d")));
    }
    #[test]
    fn test_step_expression() {
        let expr = StepExpression::<Color> {
            default_value: Color::from_hex("#f0f0f0"),
            step_values: Some(vec![
                StepValue::<Color> {
                    resolution: 10.0,
                    step_value: Color::from_hex("#fafafa"),
                },
                StepValue::<Color> {
                    resolution: 20.0,
                    step_value: Color::from_hex("#1d1d1d"),
                },
                StepValue::<Color> {
                    resolution: 30.0,
                    step_value: Color::from_hex("#1a1a1a"),
                },
            ]),
        };
        assert_eq!(expr.get_value(15.0), Some(Color::from_hex("#fafafa")));
        assert_eq!(expr.get_value(25.0), Some(Color::from_hex("#1d1d1d")));
    }
}
