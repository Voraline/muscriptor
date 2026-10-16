import { useEffect, useState } from "react";
import { Button } from "./Button";
import {
  DefaultGeneration,
  FetchGenerationDefaults,
  type Dtype,
  type GenerationSettings,
} from "../generation";

const FieldClass =
  "h-9 w-full rounded-lg border border-line-strong bg-bg px-2.5 text-sm text-content outline-none focus:border-accent disabled:opacity-50";

type NumericKey =
  | "BeamSize"
  | "CfgCoef"
  | "Temperature"
  | "TopK"
  | "TopP"
  | "BeamLengthScoreAlpha"
  | "MaxGenLen";

interface NumericSpec {
  Key: NumericKey;
  Label: string;
  Hint: string;
  Min: number;
  Max: number;
  Step: number;
  Integer: boolean;
  Needs?: "UseSampling" | "Beam";
}

const Sections: { Title: string; Fields: NumericSpec[] }[] = [
  {
    Title: "Search",
    Fields: [
      {
        Key: "BeamSize",
        Label: "Beam size",
        Hint: "1 is greedy. Higher is slower and uses more memory.",
        Min: 1,
        Max: 32,
        Step: 1,
        Integer: true,
      },
      {
        Key: "BeamLengthScoreAlpha",
        Label: "Beam length alpha",
        Hint: "Length normalization. 0 favors short sequences less strictly. Needs beam size above 1.",
        Min: 0,
        Max: 4,
        Step: 0.05,
        Integer: false,
        Needs: "Beam",
      },
      {
        Key: "CfgCoef",
        Label: "CFG coefficient",
        Hint: "Classifier-free guidance strength. 1 disables guidance.",
        Min: 0,
        Max: 10,
        Step: 0.1,
        Integer: false,
      },
      {
        Key: "MaxGenLen",
        Label: "Max tokens per chunk",
        Hint: "Generation budget for each 5 second chunk.",
        Min: 16,
        Max: 8000,
        Step: 50,
        Integer: true,
      },
    ],
  },
  {
    Title: "Sampling",
    Fields: [
      {
        Key: "Temperature",
        Label: "Temperature",
        Hint: "Lower is more conservative. Only used with sampling.",
        Min: 0.05,
        Max: 5,
        Step: 0.05,
        Integer: false,
        Needs: "UseSampling",
      },
      {
        Key: "TopK",
        Label: "Top-k",
        Hint: "Keep only the k likeliest tokens. 0 turns it off.",
        Min: 0,
        Max: 1000,
        Step: 1,
        Integer: true,
        Needs: "UseSampling",
      },
      {
        Key: "TopP",
        Label: "Top-p",
        Hint: "Nucleus cutoff between 0 and 1. 0 turns it off and wins over top-k when set.",
        Min: 0,
        Max: 1,
        Step: 0.01,
        Integer: false,
        Needs: "UseSampling",
      },
    ],
  },
];

function Field(Props: { Label: string; Hint?: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-sm text-content">{Props.Label}</span>
      {Props.children}
      {Props.Hint && <span className="text-xs text-faint">{Props.Hint}</span>}
    </label>
  );
}

function ClampNumber(Raw: string, Spec: NumericSpec, Fallback: number): number {
  const Parsed = Number(Raw);
  if (Raw.trim() === "" || !Number.isFinite(Parsed)) return Fallback;
  const Value = Spec.Integer ? Math.round(Parsed) : Parsed;
  return Math.min(Spec.Max, Math.max(Spec.Min, Value));
}

function ParseSeed(Raw: string): number | null {
  if (Raw.trim() === "") return null;
  const Parsed = Math.round(Number(Raw));
  if (!Number.isFinite(Parsed)) return null;
  return Math.min(2147483647, Math.max(0, Parsed));
}

export function GenerationPanel(Props: {
  value: GenerationSettings;
  onChange: (Next: GenerationSettings) => void;
}) {
  const { value: Value, onChange: OnChange } = Props;
  const [Open, SetOpen] = useState(false);
  const [ServerDefaults, SetServerDefaults] =
    useState<GenerationSettings>(DefaultGeneration);

  useEffect(() => {
    let Cancelled = false;
    FetchGenerationDefaults().then((D) => {
      if (Cancelled || !D) return;
      SetServerDefaults(D);
      OnChange(D);
    });
    return () => {
      Cancelled = true;
    };
  }, []);

  function Patch(P: Partial<GenerationSettings>) {
    OnChange({ ...Value, ...P });
  }

  function IsDisabled(Spec: NumericSpec): boolean {
    if (Spec.Needs === "UseSampling") return !Value.UseSampling;
    if (Spec.Needs === "Beam") return Value.BeamSize <= 1;
    return false;
  }

  return (
    <section className="card col-span-full px-5 pb-5 pt-4 animate-rise [animation-delay:0.28s]">
      <div className="flex items-start gap-3">
        <div className="flex flex-col gap-2">
          <h2 className="m-0 text-lg font-semibold">Advanced settings</h2>
          <p className="text-faint">
            Optional. Precision, search, sampling and limits for this transcription.
          </p>
        </div>
        <div className="ml-auto flex gap-2">
          <Button
            type="button"
            size="text-xs"
            pad="px-3.5 py-1"
            onClick={() => OnChange(ServerDefaults)}
          >
            Reset
          </Button>
          <Button
            type="button"
            size="text-xs"
            pad="px-3.5 py-1"
            onClick={() => SetOpen((O) => !O)}
          >
            {Open ? "Hide" : "Show"}
          </Button>
        </div>
      </div>

      {Open && (
        <div className="mt-4 flex flex-col gap-6">
          <div className="flex flex-col gap-3">
            <h3 className="m-0 text-[13px] font-semibold text-content">Precision</h3>
            <div className="grid grid-cols-2 gap-4 max-[760px]:grid-cols-1">
              <Field Label="Float type" Hint="bfloat16 needs an Ampere or newer GPU.">
                <select
                  className={FieldClass}
                  value={Value.Dtype}
                  onChange={(E) => Patch({ Dtype: E.target.value as Dtype })}
                >
                  <option value="float32">float32</option>
                  <option value="float16">float16</option>
                  <option value="bfloat16">bfloat16</option>
                </select>
              </Field>
            </div>
          </div>

          {Sections.map((Section) => (
            <div key={Section.Title} className="flex flex-col gap-3">
              <h3 className="m-0 text-[13px] font-semibold text-content">
                {Section.Title}
              </h3>
              <div className="grid grid-cols-2 gap-4 max-[760px]:grid-cols-1">
                {Section.Fields.map((Spec) => (
                  <Field key={Spec.Key} Label={Spec.Label} Hint={Spec.Hint}>
                    <input
                      className={FieldClass}
                      type="number"
                      min={Spec.Min}
                      max={Spec.Max}
                      step={Spec.Step}
                      disabled={IsDisabled(Spec)}
                      value={Value[Spec.Key]}
                      onChange={(E) =>
                        Patch({
                          [Spec.Key]: ClampNumber(
                            E.target.value,
                            Spec,
                            DefaultGeneration[Spec.Key],
                          ),
                        })
                      }
                    />
                  </Field>
                ))}
              </div>
            </div>
          ))}

          <div className="flex flex-col gap-3">
            <h3 className="m-0 text-[13px] font-semibold text-content">Behavior</h3>
            <div className="grid grid-cols-2 gap-4 max-[760px]:grid-cols-1">
              <Field
                Label="Seed"
                Hint="Leave empty for a random seed. Fix it to repeat a sampled result."
              >
                <input
                  className={FieldClass}
                  type="number"
                  min={0}
                  max={2147483647}
                  step={1}
                  placeholder="random"
                  value={Value.Seed ?? ""}
                  onChange={(E) => Patch({ Seed: ParseSeed(E.target.value) })}
                />
              </Field>
              <div className="flex flex-col justify-end gap-3">
                <label className="flex items-center gap-2 text-sm text-content">
                  <input
                    type="checkbox"
                    checked={Value.UseSampling}
                    onChange={(E) => Patch({ UseSampling: E.target.checked })}
                  />
                  Use temperature sampling instead of greedy decoding
                </label>
                <label className="flex items-center gap-2 text-sm text-content">
                  <input
                    type="checkbox"
                    checked={Value.PreludeForcing}
                    onChange={(E) => Patch({ PreludeForcing: E.target.checked })}
                  />
                  Prelude forcing (carry sounding notes across chunks)
                </label>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
