const X_min = [1.0, 1.0, 0.0, 100.0, 0.0];
const X_max = [444.0, 222.0, 100000.0, 1560780.0, 52.0];
const X_scale = [0.002257336343115124, 0.004524886877828055, 1e-05, 6.40746341338391e-07, 0.019230769230769232];

const Y_min = 1.0;
const Y_max = 515000000.0;
const Y_scale = 1.9417475765859177e-09;

let model;

(async function loadModel() {
    try {
        console.log('Starting model loading...');
        console.log('Attempting to load from:', '/Models/real_estate_tfjs_model/model.json');

        model = await tf.loadGraphModel('/Models/real_estate_tfjs_model/model.json', { fromTFHub: false });
        console.log('Model loaded:', model);

        if (!model) throw new Error('Model loaded but is undefined');

        console.log('Model loaded successfully');
        console.log('Model input shape:', model.inputs[0].shape);

        const testTensor = tf.zeros([1, 5]);
        const testPrediction = model.predict(testTensor);
        console.log('Test prediction successful:', testPrediction !== null);
        testTensor.dispose();
        testPrediction.dispose();
    } catch (error) {
        console.error('Error loading model:', {
            message: error.message,
            stack: error.stack,
            modelPath: '/tfjs_model/model.json'
        });

        if (error.message.includes('404') || error.message.includes('Failed to fetch')) {
            alert('Model file not found. Please check if the model files are in the correct location.');
        }
    }
})();

const stateMapping = {
    "Alabama": 0, "Alaska": 1, "Arizona": 2, "Arkansas": 3, "California": 4,
    "Colorado": 5, "Connecticut": 6, "Delaware": 7, "District of Columbia": 8, "Florida": 9,
    "Georgia": 10, "Hawaii": 11, "Idaho": 12, "Illinois": 13, "Indiana": 14,
    "Iowa": 15, "Kansas": 16, "Kentucky": 17, "Louisiana": 18, "Maine": 19,
    "Maryland": 20, "Massachusetts": 21, "Michigan": 22, "Minnesota": 23, "Mississippi": 24,
    "Missouri": 25, "Montana": 26, "Nebraska": 27, "Nevada": 28, "New Hampshire": 29,
    "New Jersey": 30, "New Mexico": 31, "New York": 32, "North Carolina": 33, "North Dakota": 34,
    "Ohio": 35, "Oklahoma": 36, "Oregon": 37, "Pennsylvania": 38, "Puerto Rico": 39,
    "Rhode Island": 40, "South Carolina": 41, "South Dakota": 42, "Tennessee": 43, "Texas": 44,
    "Utah": 45, "Vermont": 46, "Virgin Islands": 47, "Virginia": 48, "Washington": 49,
    "West Virginia": 50, "Wisconsin": 51, "Wyoming": 52
};

function normalizeInput(input) {
    return input.map((value, index) => (value - X_min[index]) * X_scale[index]);
}

function deNormalizePrediction(scaledPrediction) {
    return scaledPrediction / Y_scale + Y_min;
}

document.getElementById('prediction-form').addEventListener('submit', async (event) => {
    event.preventDefault();

    if (!model) {
        alert('Model is still loading. Please wait a moment and try again.');
        return;
    }

    const bed = parseFloat(document.getElementById('bed').value);
    const bath = parseFloat(document.getElementById('bath').value);
    const acreLot = parseFloat(document.getElementById('acre_lot').value);
    const houseSize = parseFloat(document.getElementById('house_size').value);
    const state = stateMapping[document.getElementById('state').value];

    if (typeof state === 'undefined') {
        alert('Please select a valid state.');
        return;
    }

    if ([bed, bath, acreLot, houseSize].some(v => isNaN(v) || v < 0)) {
        alert('Please enter valid positive numbers for all fields.');
        return;
    }

    const input = [bed, bath, acreLot, houseSize, state];
    const normalizedInput = normalizeInput(input);
    console.log('Normalized Input:', normalizedInput);

    const inputTensor = tf.tensor2d([normalizedInput], [1, 5]);
    const prediction = model.predict(inputTensor);

    if (prediction) {
        const scaledPrediction = (await prediction.array())[0][0];
        console.log('Raw Scaled Prediction:', scaledPrediction);

        const finalPrice = deNormalizePrediction(scaledPrediction);
        console.log('Final Price:', finalPrice);

        document.getElementById('predicted-price').textContent = finalPrice.toFixed(3);
    } else {
        console.error('Prediction failed: model returned null or undefined');
    }

    inputTensor.dispose();
    prediction.dispose();
});
