export type Dtype = "float32" | "float16" | "bfloat16";

export interface GenerationSettings {
  Dtype: Dtype;
  BeamSize: number;
  UseSampling: boolean;
  Temperature: number;
  CfgCoef: number;
  PreludeForcing: boolean;
  MaxGenLen: number;
  TopK: number;
  TopP: number;
  BeamLengthScoreAlpha: number;
  Seed: number | null;
}

export const DefaultGeneration: GenerationSettings = {
  Dtype: "float32",
  BeamSize: 1,
  UseSampling: false,
  Temperature: 1,
  CfgCoef: 1,
  PreludeForcing: true,
  MaxGenLen: 2000,
  TopK: 0,
  TopP: 0,
  BeamLengthScoreAlpha: 0.75,
  Seed: null,
};

export function GenerationToForm(G: GenerationSettings): Record<string, string> {
  const Form: Record<string, string> = {
    dtype: G.Dtype,
    beam_size: String(G.BeamSize),
    use_sampling: String(G.UseSampling),
    cfg_coef: String(G.CfgCoef),
    prelude_forcing: String(G.PreludeForcing),
    max_gen_len: String(G.MaxGenLen),
  };
  if (G.UseSampling) {
    Form.temperature = String(G.Temperature);
    Form.top_k = String(G.TopK);
    Form.top_p = String(G.TopP);
  }
  if (G.BeamSize > 1) Form.beam_length_score_alpha = String(G.BeamLengthScoreAlpha);
  if (G.Seed !== null) Form.seed = String(G.Seed);
  return Form;
}

export async function FetchGenerationDefaults(): Promise<GenerationSettings | null> {
  try {
    const Resp = await fetch("/settings");
    if (!Resp.ok) return null;
    const D = await Resp.json();
    return {
      Dtype: D.dtype,
      BeamSize: D.beam_size,
      UseSampling: D.use_sampling,
      Temperature: D.temperature,
      CfgCoef: D.cfg_coef,
      PreludeForcing: D.prelude_forcing,
      MaxGenLen: D.max_gen_len ?? DefaultGeneration.MaxGenLen,
      TopK: D.top_k ?? DefaultGeneration.TopK,
      TopP: D.top_p ?? DefaultGeneration.TopP,
      BeamLengthScoreAlpha:
        D.beam_length_score_alpha ?? DefaultGeneration.BeamLengthScoreAlpha,
      Seed: D.seed ?? null,
    };
  } catch {
    return null;
  }
}
