import { Value } from "./engine";
import * as tf from "@tensorflow/tfjs";

// These tests are translated from https://github.com/karpathy/micrograd/blob/master/test/test_engine.py
describe("class Value", () => {
  describe("sanity check", () => {
    function f(x) {
      let z = x.mul(2).add(2).add(x);
      let q = z.relu().add(z.mul(x));
      let h = z.mul(z).relu();
      return h.add(q).add(q.mul(x));
    }
    let xmg = new Value(-4);
    let ymg = f(xmg);
    ymg.backward();

    const g = tf.valueAndGrad(f);
    const { value, grad } = g(tf.tensor(-4));

    test("forward pass", () =>
      expect(ymg.data).toBeCloseTo(value.dataSync()[0]));
    test("backward pass", () =>
      expect(xmg.grad).toBeCloseTo(grad.dataSync()[0]));
  });
  describe("test more ops", () => {
    function f(a, b) {
      let c = a.add(b);
      let d = a.mul(b).add(b.pow(3));
      c = c.add(c.add(1));
      c = c.add(c.add(1).add(c).add(a.mul(-1)));
      d = d.add(d.mul(2).add(b.add(a).relu()));
      d = d.add(d.add(d.mul(3).add(b.sub(a).relu())));
      let e = c.sub(d);
      let f = e.pow(2);
      let g = f.div(2);
      return g.add(a.div(f));
    }

    let amg = new Value(-4);
    let bmg = new Value(2);
    let gmg = f(amg, bmg);
    gmg.backward();

    const g_a = tf.valueAndGrad((a) => f(a, tf.tensor(2)));
    const g_b = tf.grad((b) => f(tf.tensor(-4), b));

    const { value, grad } = g_a(tf.tensor(-4));
    const grad_b = g_b(tf.tensor(2));

    test("forward pass", () =>
      expect(gmg.data).toBeCloseTo(value.dataSync()[0]));
    describe("backward pass", () => {
      test("w.r.t. a", () => expect(amg.grad).toBeCloseTo(grad.dataSync()[0]));
      test("w.r.t. b", () =>
        expect(bmg.grad).toBeCloseTo(grad_b.dataSync()[0]));
    });
  });
});
