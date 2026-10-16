import { useEffect, useState } from "react";
import { Button } from "./Button";
import {
  DEFAULT_GENERATION,
  fetchGenerationDefaults,
  type Dtype,
  type GenerationSettings,
} from "../generation";

const fieldClass =
  "h-9 w-full rounded-lg border border-line-strong bg-bg px-2.5 text-sm text-content outline-none focus:border-accent";

function Field(props: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-sm text-content">{props.label}</span>
      {props.children}
      {props.hint && <span className="text-xs text-faint">{props.hint}</span>}
    </label>
  );
}

export function GenerationPanel(props: {
  value: GenerationSettings;
  onChange: (next: GenerationSettings) => void;
}) {
  const { value, onChange } = props;
  const [open, setOpen] = useState(false);
  const [serverDefaults, setServerDefaults] = useState<GenerationSettings>(
    DEFAULT_GENERATION,
  );

  useEffect(() => {
    let cancelled = false;
    fetchGenerationDefaults().then((d) => {
      if (cancelled || !d) return;
      setServerDefaults(d);
      onChange(d);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  function patch(p: Partial<GenerationSettings>) {
    onChange({ ...value, ...p });
  }

  return (
    <section className="card col-span-full px-5 pb-5 pt-4 animate-rise [animation-delay:0.28s]">
      <div className="flex items-start gap-3">
        <div className="flex flex-col gap-2">
          <h2 className="m-0 text-lg font-semibold">Advanced settings</h2>
          <p className="text-faint">
            Optional. Precision, beam search and sampling for this transcription.
          </p>
        </div>
        <div className="ml-auto flex gap-2">
          <Button
            type="button"
            size="text-xs"
            pad="px-3.5 py-1"
            onClick={() => onChange(serverDefaults)}
          >
            Reset
          </Button>
          <Button
            type="button"
            size="text-xs"
            pad="px-3.5 py-1"
            onClick={() => setOpen((o) => !o)}
          >
            {open ? "Hide" : "Show"}
          </Button>
        </div>
      </div>

      {open && (
        <div className="mt-4 grid grid-cols-2 gap-4 max-[760px]:grid-cols-1">
          <Field label="Float type" hint="bfloat16 needs an Ampere or newer GPU.">
            <select
              className={fieldClass}
              value={value.dtype}
              onChange={(e) => patch({ dtype: e.target.value as Dtype })}
            >
              <option value="float32">float32</option>
              <option value="float16">float16</option>
              <option value="bfloat16">bfloat16</option>
            </select>
          </Field>

          <Field label="Beam size" hint="1 is greedy. Higher is slower and uses more memory.">
            <input
              className={fieldClass}
              type="number"
              min={1}
              max={32}
              step={1}
              value={value.beamSize}
              onChange={(e) =>
                patch({ beamSize: Math.min(32, Math.max(1, Math.round(Number(e.target.value) || 1))) })
              }
            />
          </Field>

          <Field label="CFG coefficient" hint="Classifier-free guidance strength.">
            <input
              className={fieldClass}
              type="number"
              min={0}
              step={0.1}
              value={value.cfgCoef}
              onChange={(e) => patch({ cfgCoef: Number(e.target.value) })}
            />
          </Field>

          <Field label="Temperature" hint="Only used with sampling.">
            <input
              className={fieldClass}
              type="number"
              min={0.05}
              step={0.05}
              disabled={!value.useSampling}
              value={value.temperature}
              onChange={(e) => patch({ temperature: Number(e.target.value) })}
            />
          </Field>

          <label className="flex items-center gap-2 text-sm text-content">
            <input
              type="checkbox"
              checked={value.useSampling}
              onChange={(e) => patch({ useSampling: e.target.checked })}
            />
            Use temperature sampling instead of greedy decoding
          </label>

          <label className="flex items-center gap-2 text-sm text-content">
            <input
              type="checkbox"
              checked={value.preludeForcing}
              onChange={(e) => patch({ preludeForcing: e.target.checked })}
            />
            Prelude forcing (carry sounding notes across chunks)
          </label>
        </div>
      )}
    </section>
  );
}
