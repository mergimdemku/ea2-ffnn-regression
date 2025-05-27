// FFNN Regression Task - TensorFlow.js
let rawData = [], noiseData = [];
let model, modelBest, modelOverfit;

async function generateData(N = 100, noise = false) {
  const x = Array.from({ length: N }, () => Math.random() * 4 - 2);
  const y = x.map(v => 0.5 * (v + 0.8) * (v + 1.8) * (v - 0.2) * (v - 0.3) * (v - 1.9) + 1);
  let data = x.map((x, i) => ({ x, y: noise ? y[i] + gaussianNoise(0, 0.05) : y[i] }));
  return data;
}

function gaussianNoise(mean = 0, stdDev = 1) {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return stdDev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) + mean;
}

function splitData(data) {
  tf.util.shuffle(data);
  const split = Math.floor(data.length / 2);
  return [data.slice(0, split), data.slice(split)];
}

function createModel() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ inputShape: [1], units: 100, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 100, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 1 }));
  m.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
  return m;
}

async function trainModel(model, train, epochs = 50) {
  const xs = tf.tensor2d(train.map(d => [d.x]));
  const ys = tf.tensor2d(train.map(d => [d.y]));
  const history = await model.fit(xs, ys, {
    batchSize: 32,
    epochs,
    shuffle: true,
    verbose: 0
  });
  xs.dispose(); ys.dispose();
  return history;
}

async function evaluate(model, data) {
  const xs = tf.tensor2d(data.map(d => [d.x]));
  const ys = tf.tensor2d(data.map(d => [d.y]));
  const loss = await model.evaluate(xs, ys);
  xs.dispose(); ys.dispose();
  return (await loss.data())[0];
}

async function predict(model, data) {
  const xs = tf.tensor2d(data.map(d => [d.x]));
  const ys = model.predict(xs);
  const predictions = await ys.array();
  xs.dispose(); ys.dispose();
  return predictions.map((y, i) => ({ x: data[i].x, y: y[0] }));
}

function plot(canvasId, data, label, color) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [{
        label,
        data: data.map(p => ({ x: p.x, y: p.y })),
        borderColor: color,
        backgroundColor: color,
        showLine: false,
        pointRadius: 3
      }]
    },
    options: {
      scales: {
        x: { type: 'linear', position: 'bottom' },
        y: { beginAtZero: false }
      }
    }
  });
}
