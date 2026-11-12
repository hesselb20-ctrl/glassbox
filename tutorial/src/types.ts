// Shape of src/data.json (written by make_data.py). Every field is optional so a partial file still renders.
export interface ModAdd {
  available?: boolean;
  p?: number;
  frequencies?: number[];
  share?: number[];
  key_frequencies?: number[];
  top?: {frequency: number; share: number}[];
  hypothesis?: string;
  result?: string;
  revised_example?: {text?: string; target?: string};
  pseudocode?: string;
  verification?: {
    n?: number;
    agreement?: number;
    subspace_dims?: number;
    total_dims?: number;
    acc_only_subspace?: number;
    acc_subspace_removed?: number;
    acc_random_removed?: number;
  };
  experiments?: number;
  wall_clock_seconds?: number;
  tag?: string;
}

export interface ScoreRow {
  name?: string;
  title?: string;
  found?: number;
  partial?: number;
  missed?: number;
  total?: number;
  experiments?: number;
  minutes?: number;
  status?: string;
}

export interface Demo27b {
  available?: boolean;
  model?: string;
  label?: string;
  n_layers?: number;
  separation_by_layer?: number[];
  best_layer?: number;
  steer_layers?: number[];
  steering?: {alphas?: number[]; metric?: number[]; metric_name?: string};
  generations?: {prompt?: string; before?: string; after?: string; alpha?: number}[];
  gpus?: string[];
  tag?: string;
}

export interface Rung {
  model?: string;
  params?: string | number;
  backend?: string;
  tag?: string;
  status?: string;
  key_result?: string;
  heads?: string[];
}

export interface Data {
  modadd?: ModAdd;
  scorecard?: {
    available?: boolean;
    rows?: ScoreRow[];
    ioi?: {found?: number; published?: number};
  };
  demo27b?: Demo27b;
  ladder?: {available?: boolean; behavior?: string; rungs?: Rung[]; conclusion?: string};
  repo?: string;
  install?: string;
}

export interface SceneProps {
  data: Data;
  dur: number;
  frame?: number; // overrides the timeline frame (Preview time remapping)
}
