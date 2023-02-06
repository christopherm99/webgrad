export class Value {
  public data: number;
  public grad: number;

  private _backward: () => void;
  private _prev: Set<Value>;
  private _op: string;

  constructor(data:number, _children:Set<Value> = new Set(), _op:string='') {
    this.data = data;
    this.grad = 0;
    // Internal variables
    this._backward = () => {};
    this._prev = _children;
    this._op = _op;
  }

  add(_other: number | Value) {
    let other = typeof _other === "number" ? new Value(_other) : _other;
    let out = new Value(this.data + other.data, new Set([this, other]), "+");

    function _backward() {
      this.grad += out.grad;
      other.grad += out.grad;
    }

    out._backward = _backward;

    return out;
  }


  mul(_other: number | Value) {
    let other = typeof _other === "number" ? new Value(_other) : _other;
    let out = new Value(this.data * other.data, new Set([this, other]), "*");

    function _backward() {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    }

    out._backward = _backward;

    return out;
  }

  pow(other: number) {
    let out = new Value(Math.pow(this.data, other), new Set([this]), `**${other}`);

    function _backward() {
      this.grad += (other * Math.pow(this.data, other-1)) * out.grad;
    }

    out._backward = _backward;

    return out;
  }

  relu() {
    let out = new Value(this.data < 0 ? 0 : this.data, new Set([this]), "ReLU");

    function _backward() {
      this.grad += (out.data < 0 ? 0 : 1) * out.grad; // Derivative of ReLU at zero is 1
    }

    out._backward = _backward;

    return out;
  }

  backward() {
    let topo: Array<Value> = []
    let visited: Set<Value> = new Set();
    function build_topo(v: Value) {
      if (!visited.has(v)) {
        visited.add(v);
        v._prev.forEach(build_topo);
        topo.push(v)
      }
    }
    build_topo(this);

    this.grad = 1
    topo.reverse().forEach(v => v._backward());
  }

  sub(other: number | Value) {
    return this.add((typeof other === "number" ? new Value(other) : other).mul(-1));
  }

  div(other: number | Value) {
    return this.mul((typeof other === "number" ? new Value(other) : other).pow(-1));
  }

  toString() {
    return `Value(data=${this.data}, grad=${this.grad})`;
  }
}
