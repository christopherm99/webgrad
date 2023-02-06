import { Value } from "./engine";
import { randomUniform } from "./utils";

export class Module {
  parameters(): Array<Value> {
    return [];
  }

  zero_grad(): void {
    this.parameters().forEach((p) => (p.grad = 0));
  }
}

class Neuron extends Module {
  public w: Array<Value>;
  public b: Value;

  nonlin: boolean;

  constructor(nin: number, nonlin: boolean = true) {
    super();
    this.w = Array(nin).map(() => new Value(randomUniform(-1, 1)));
    this.b = new Value(0);
    this.nonlin = nonlin;
  }

  apply(x: Array<Value>) {
    let act = this.w
      .map((wi, i) => wi.mul(x[i]))
      .reduce((acc, w) => w.add(acc))
      .add(this.b);
    return this.nonlin ? act.relu() : act;
  }

  parameters(): Array<Value> {
    return this.w.concat(this.b);
  }

  toString(): string {
    return `${this.nonlin ? "ReLu" : "Linear"}Neuron(${this.w.length})`;
  }
}

class Layer extends Module {
  neurons: Array<Neuron>;

  constructor(nin: number, nout: number, ...kwargs) {
    super();
    this.neurons = Array(nin).map(() => new Neuron(nin, ...kwargs));
  }

  apply(x) {
    let out = this.neurons.map((n) => n.apply(x));
    return out.length === 1 ? out[0] : out;
  }

  parameters(): Array<Value> {
    return this.neurons.map((n) => n.parameters()).flat();
  }

  toString(): string {
    return `Layer of [${this.neurons.map((n) => n.toString()).join(", ")}]`;
  }
}

export class MLP extends Module {
  layers: Array<Layer>;

  constructor(nin: number, nouts: Array<number>) {
    super();
    let sz = [nin].concat(nouts);
    this.layers = nouts.map(
      (_, i) => new Layer(sz[i], sz[i + 1], i != nouts.length - 1)
    );
  }

  apply(x) {
    this.layers.forEach((layer) => (x = layer.apply(x)));
    return x;
  }

  parameters(): Array<Value> {
    return this.layers.map((layer) => layer.parameters()).flat();
  }

  toString(): string {
    return `MLP of [${this.layers.map((l) => l.toString()).join(", ")}]`;
  }
}
