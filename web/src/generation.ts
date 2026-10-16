export type Dtype = "float32" | "float16" | "bfloat16";

export interface GenerationSettings {
  dtype: Dtype;
  beamSize: number;
  useSampling: boolean;
  temperature: number;
  cfgCoef: number;
  preludeForcing: boolean;
}

export const DEFAULT_GENERATION: GenerationSettings = {
  dtype: "float32",
  beamSize: 1,
  useSampling: false,
  temperature: 1,
  cfgCoef: 1,
  preludeForcing: true,
};

export function generationToForm(g: GenerationSettings): Record<string, string> {
  const form: Record<string, string> = {
    dtype: g.dtype,
    beam_size: String(g.beamSize),
    use_sampling: String(g.useSampling),
    cfg_coef: String(g.cfgCoef),
    prelude_forcing: String(g.preludeForcing),
  };
  if (g.useSampling) form.temperature = String(g.temperature);
  return form;
}

export async function fetchGenerationDefaults(): Promise<GenerationSettings | null> {
  try {
    const resp = await fetch("/settings");
    if (!resp.ok) return null;
    const d = await resp.json();
    return {
      dtype: d.dtype,
      beamSize: d.beam_size,
      useSampling: d.use_sampling,
      temperature: d.temperature,
      cfgCoef: d.cfg_coef,
      preludeForcing: d.prelude_forcing,
    };
  } catch {
    return null;
  }
}
