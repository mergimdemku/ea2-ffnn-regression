
// ffnn_regression_plotly.js
// Starterpaket: R1–R4 mit Plotly & TensorFlow.js

// 1. Hilfsfunktionen: Daten erzeugen & Rauschen
function gaussianNoise(mean = 0, stdDev = 1) {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return stdDev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) + mean;
}

function generateData(N = 100, noise = false) {
  const x = Array.from({ length: N }, () => Math.random() * 4 - 2);
  const y = x.map(v => 0.5 * (v + 0.8) * (v + 1.8) * (v - 0.2) * (v - 0.3) * (v - 1.9) + 1);
  return x.map((xi, i) => ({ x: xi, y: noise ? y[i] + gaussianNoise(0, Math.sqrt(0.05)) : y[i] }));
}

function splitData(data) {
  tf.util.shuffle(data);
  const split = Math.floor(data.length / 2);
  return [data.slice(0, split), data.slice(split)];
}

// 2. Modell erstellen
function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [1], units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));
  model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
  return model;
}

async function trainAndPredict(model, trainData, testData, epochs = 100) {
  const trainXs = tf.tensor2d(trainData.map(d => [d.x]));
  const trainYs = tf.tensor2d(trainData.map(d => [d.y]));
  await model.fit(trainXs, trainYs, { epochs, batchSize: 32, shuffle: true, verbose: 0 });
  const trainPred = await model.predict(trainXs).array();
  const testXs = tf.tensor2d(testData.map(d => [d.x]));
  const testPred = await model.predict(testXs).array();
  const trainLoss = await model.evaluate(trainXs, trainYs).data();
  const testYs = tf.tensor2d(testData.map(d => [d.y]));
  const testLoss = await model.evaluate(testXs, testYs).data();
  return {
    trainPred: trainData.map((d, i) => ({ x: d.x, y: trainPred[i][0] })),
    testPred: testData.map((d, i) => ({ x: d.x, y: testPred[i][0] })),
    lossTrain: trainLoss[0],
    lossTest: testLoss[0]
  };
}

function plotData(divId, points, title, color = 'blue') {
  Plotly.newPlot(divId, [{
    x: points.map(p => p.x),
    y: points.map(p => p.y),
    mode: 'markers',
    type: 'scatter',
    marker: { color }
  }], { title });
}

function plotPrediction(divId, truePoints, predPoints, title) {
  Plotly.newPlot(divId, [
    {
      x: truePoints.map(p => p.x),
      y: truePoints.map(p => p.y),
      mode: 'markers',
      type: 'scatter',
      name: 'True',
      marker: { color: 'gray' }
    },
    {
      x: predPoints.map(p => p.x),
      y: predPoints.map(p => p.y),
      mode: 'lines',
      type: 'scatter',
      name: 'Prediction',
      line: { color: 'red' }
    }
  ], { title });
}

// 3. Hauptlogik – alle vier Modelle erzeugen und anzeigen
window.onload = async () => {
  const cleanData = generateData(100, false);
  const noisyData = generateData(100, true);
  const [trainClean, testClean] = splitData(cleanData);
  const [trainNoisy, testNoisy] = splitData(noisyData);

  plotData('dataset-plot-clean', cleanData, 'R1: Clean Data', 'blue');
  plotData('dataset-plot-noisy', noisyData, 'R1: Noisy Data', 'orange');

  const modelClean = createModel();
  const resultClean = await trainAndPredict(modelClean, trainClean, testClean, 100);
  plotPrediction('prediction-clean-train', trainClean, resultClean.trainPred, 'R2: Clean Train');
  plotPrediction('prediction-clean-test', testClean, resultClean.testPred, 'R2: Clean Test');
  document.getElementById('loss-clean').innerText = `Train Loss: ${resultClean.lossTrain.toFixed(4)}, Test Loss: ${resultClean.lossTest.toFixed(4)}`;

  const modelBest = createModel();
  const resultBest = await trainAndPredict(modelBest, trainNoisy, testNoisy, 100);
  plotPrediction('prediction-best-train', trainNoisy, resultBest.trainPred, 'R3: Best-Fit Train');
  plotPrediction('prediction-best-test', testNoisy, resultBest.testPred, 'R3: Best-Fit Test');
  document.getElementById('loss-best').innerText = `Train Loss: ${resultBest.lossTrain.toFixed(4)}, Test Loss: ${resultBest.lossTest.toFixed(4)}`;

  const modelOverfit = createModel();
  const resultOverfit = await trainAndPredict(modelOverfit, trainNoisy, testNoisy, 500);
  plotPrediction('prediction-overfit-train', trainNoisy, resultOverfit.trainPred, 'R4: Overfit Train');
  plotPrediction('prediction-overfit-test', testNoisy, resultOverfit.testPred, 'R4: Overfit Test');
  document.getElementById('loss-overfit').innerText = `Train Loss: ${resultOverfit.lossTrain.toFixed(4)}, Test Loss: ${resultOverfit.lossTest.toFixed(4)}`;
};
