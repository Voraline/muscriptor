import { useEffect, useRef, useState } from "react";
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

// Returns null when the text isn't a usable number yet (empty, "-", "1e", ...).
function ClampNumber(Raw: string, Spec: NumericSpec): number | null {
  const Parsed = Number(Raw);
  if (Raw.trim() === "" || !Number.isFinite(Parsed)) return null;
  const Value = Spec.Integer ? Math.round(Parsed) : Parsed;
  return Math.min(Spec.Max, Math.max(Spec.Min, Value));
}

// Keeps the raw text while the user types so partial input like "0" or "5"
// is not snapped to the minimum mid-edit. The value is clamped on blur, and
// valid in-progress input is still pushed up immediately.
function NumericField(Props: {
  Spec: NumericSpec;
  Value: number;
  Fallback: number;
  Disabled: boolean;
  OnCommit: (Next: number) => void;
}) {
  const { Spec, Value, Fallback, Disabled, OnCommit } = Props;
  const [Draft, SetDraft] = useState(String(Value));
  const Focused = useRef(false);

  // Follow external changes (Reset, server defaults) unless the user is typing.
  useEffect(() => {
    if (!Focused.current) SetDraft(String(Value));
  }, [Value]);

  return (
    <input
      className={FieldClass}
      type="number"
      min={Spec.Min}
      max={Spec.Max}
      step={Spec.Step}
      disabled={Disabled}
      value={Draft}
      onFocus={() => {
        Focused.current = true;
      }}
      onChange={(E) => {
        SetDraft(E.target.value);
        const Next = ClampNumber(E.target.value, Spec);
        if (Next !== null) OnCommit(Next);
      }}
      onBlur={(E) => {
        Focused.current = false;
        const Next = ClampNumber(E.target.value, Spec) ?? Fallback;
        OnCommit(Next);
        SetDraft(String(Next));
      }}
    />
  );
}

// Survives the panel unmounting/remounting within one page load.
let CachedServerDefaults: GenerationSettings | null = null;

function SameSettings(A: GenerationSettings, B: GenerationSettings): boolean {
  return (Object.keys(A) as (keyof GenerationSettings)[]).every((K) => A[K] === B[K]);
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

  const ValueRef = useRef(Value);
  ValueRef.current = Value;
  const OnChangeRef = useRef(OnChange);
  OnChangeRef.current = OnChange;

  useEffect(() => {
    // Already fetched once this page load: reuse it and leave the user's
    // current settings alone (the panel remounts when the screen changes).
    if (CachedServerDefaults) {
      SetServerDefaults(CachedServerDefaults);
      return;
    }
    let Cancelled = false;
    FetchGenerationDefaults().then((D) => {
      if (!D) return;
      CachedServerDefaults = D;
      if (Cancelled) return;
      SetServerDefaults(D);
      // Only adopt the server defaults if the user hasn't changed anything yet.
      if (SameSettings(ValueRef.current, DefaultGeneration)) OnChangeRef.current(D);
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
                    <NumericField
                      Spec={Spec}
                      Value={Value[Spec.Key]}
                      Fallback={ServerDefaults[Spec.Key]}
                      Disabled={IsDisabled(Spec)}
                      OnCommit={(Next) => Patch({ [Spec.Key]: Next })}
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
