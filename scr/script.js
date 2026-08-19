// Client-side K-Means & Elbow Method Implementation in JavaScript

let chartInstance = null;
let currentImageData = null; // Original full-resolution image object
let currentPreset = 'imagen1';

// Preset configurations
const presets = {
    'imagen1': {
        name: 'Neon Cyberpunk',
        url: 'data/imagen1_reducido.jpg',
        resolution: '1024 x 1024',
        defaultK: 4
    },
    'imagen2': {
        name: 'Playa Atardecer',
        url: 'data/imagen2_reducido.jpg',
        resolution: '1024 x 1024',
        defaultK: 5
    }
};

// Initial setup on page load
window.addEventListener('DOMContentLoaded', () => {
    // Load default preset (imagen1)
    loadPreset('imagen1');
});

// Load preset images
function loadPreset(presetKey) {
    currentPreset = presetKey;
    const preset = presets[presetKey];
    
    // Update active button state
    document.getElementById('btn-preset-1').classList.toggle('active', presetKey === 'imagen1');
    document.getElementById('btn-preset-2').classList.toggle('active', presetKey === 'imagen2');
    
    // Clear custom file upload styling if active
    document.getElementById('image-upload').value = '';
    
    // Load image
    const img = new Image();
    img.crossOrigin = 'anonymous'; // Allow reading pixels if hosted externally
    img.src = preset.url;
    img.onload = () => {
        setupLoadedImage(img, preset.resolution);
        
        // Auto run the first analysis for immediate satisfaction
        runAnalysis();
    };
}

// Handle custom image upload
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Reset preset buttons styling
    document.getElementById('btn-preset-1').classList.remove('active');
    document.getElementById('btn-preset-2').classList.remove('active');
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.src = e.target.result;
        img.onload = () => {
            setupLoadedImage(img, `${img.width} x ${img.height}`);
            // Auto run analysis on uploaded image
            runAnalysis();
        };
    };
    reader.readAsDataURL(file);
}

// Setup original and hidden canvas with the loaded image
function setupLoadedImage(img, resolutionStr) {
    currentImageData = img;
    document.getElementById('display-original').src = img.src;
    document.getElementById('orig-resolution').textContent = resolutionStr;
    
    // Setup hidden canvas used for downsampling
    const canvasHidden = document.getElementById('canvas-hidden');
    const ctxHidden = canvasHidden.getContext('2d');
    
    // Downsample size (50x50 for rapid client-side clustering)
    canvasHidden.width = 50;
    canvasHidden.height = 50;
    ctxHidden.drawImage(img, 0, 0, 50, 50);
}

// K-Means algorithm implementation in JavaScript
class KMeans {
    constructor(k, maxIterations = 15) {
        this.k = k;
        this.maxIterations = maxIterations;
        this.centroids = [];
    }

    // Initialize centroids using random selection
    initializeCentroids(data) {
        this.centroids = [];
        const usedIndices = new Set();
        while (this.centroids.length < this.k) {
            const index = Math.floor(Math.random() * data.length);
            if (!usedIndices.has(index)) {
                usedIndices.add(index);
                this.centroids.push([...data[index]]); // Deep copy pixel RGB
            }
        }
    }

    // Euclidean distance in RGB space
    distance(color1, color2) {
        return Math.sqrt(
            Math.pow(color1[0] - color2[0], 2) +
            Math.pow(color1[1] - color2[1], 2) +
            Math.pow(color1[2] - color2[2], 2)
        );
    }

    fit(data) {
        this.initializeCentroids(data);
        let iterations = 0;
        let converged = false;

        while (iterations < this.maxIterations && !converged) {
            const clusters = Array.from({ length: this.k }, () => []);
            
            // Assign step
            for (let i = 0; i < data.length; i++) {
                const pixel = data[i];
                let minDistance = Infinity;
                let clusterIndex = 0;

                for (let j = 0; j < this.k; j++) {
                    const dist = this.distance(pixel, this.centroids[j]);
                    if (dist < minDistance) {
                        minDistance = dist;
                        clusterIndex = j;
                    }
                }
                clusters[clusterIndex].push(pixel);
            }

            // Update step
            const newCentroids = [];
            converged = true;

            for (let j = 0; j < this.k; j++) {
                const cluster = clusters[j];
                if (cluster.length === 0) {
                    // Reinitialize empty cluster centroid to a random pixel
                    newCentroids.push([...data[Math.floor(Math.random() * data.length)]]);
                    converged = false;
                } else {
                    const sum = cluster.reduce(
                        (acc, pixel) => [acc[0] + pixel[0], acc[1] + pixel[1], acc[2] + pixel[2]],
                        [0, 0, 0]
                    );
                    const mean = [
                        sum[0] / cluster.length,
                        sum[1] / cluster.length,
                        sum[2] / cluster.length
                    ];

                    // Check convergence (did centroid move significantly?)
                    if (this.distance(mean, this.centroids[j]) > 0.5) {
                        converged = false;
                    }
                    newCentroids.push(mean);
                }
            }

            this.centroids = newCentroids;
            iterations++;
        }

        // Calculate SSE (Within-Cluster Sum of Squares)
        let sse = 0;
        for (let i = 0; i < data.length; i++) {
            const pixel = data[i];
            let minDistanceSq = Infinity;
            for (let j = 0; j < this.k; j++) {
                const distSq = 
                    Math.pow(pixel[0] - this.centroids[j][0], 2) +
                    Math.pow(pixel[1] - this.centroids[j][1], 2) +
                    Math.pow(pixel[2] - this.centroids[j][2], 2);
                if (distSq < minDistanceSq) {
                    minDistanceSq = distSq;
                }
            }
            sse += minDistanceSq;
        }

        return { centroids: this.centroids, sse: sse };
    }

    // Map pixel list to nearest centroids
    predict(pixel) {
        let minDistance = Infinity;
        let index = 0;
        for (let j = 0; j < this.k; j++) {
            const dist = this.distance(pixel, this.centroids[j]);
            if (dist < minDistance) {
                minDistance = dist;
                index = j;
            }
        }
        return this.centroids[index];
    }
}

// Find the Elbow Point using distance-from-chord method
function detectElbowPoint(kValues, sseValues) {
    const x1 = kValues[0];
    const y1 = sseValues[0];
    const x2 = kValues[kValues.length - 1];
    const y2 = sseValues[sseValues.length - 1];

    // Line representation Ax + By + C = 0
    const A = y1 - y2;
    const B = x2 - x1;
    const C = x1 * y2 - x2 * y1;

    let maxDistance = -1;
    let elbowK = kValues[0];

    for (let i = 0; i < kValues.length; i++) {
        const x = kValues[i];
        const y = sseValues[i];
        // Perpendicular distance from point to line
        const distance = Math.abs(A * x + B * y + C) / Math.sqrt(A * A + B * B);
        if (distance > maxDistance) {
            maxDistance = distance;
            elbowK = x;
        }
    }
    return elbowK;
}

// Main execution routine
function runAnalysis() {
    if (!currentImageData) return;

    // Show loading spinner
    const loader = document.getElementById('loader');
    loader.classList.remove('opacity-0', 'pointer-events-none');
    
    // Use setTimeout to allow browser to draw loader before heavy calculations
    setTimeout(() => {
        try {
            const canvasHidden = document.getElementById('canvas-hidden');
            const ctxHidden = canvasHidden.getContext('2d');
            const imgData = ctxHidden.getImageData(0, 0, 50, 50).data;

            // Flatten image data into RGB pixel list [r, g, b]
            const pixels = [];
            for (let i = 0; i < imgData.length; i += 4) {
                pixels.push([imgData[i], imgData[i+1], imgData[i+2]]);
            }

            const kValues = [2, 3, 4, 5, 6, 7, 8, 9, 10];
            const sseValues = [];
            const resultsByK = {};

            // Run KMeans for K = 2..10
            kValues.forEach(k => {
                const kmeans = new KMeans(k);
                const result = kmeans.fit(pixels);
                sseValues.push(result.sse);
                resultsByK[k] = result;
            });

            // Find optimal K
            const optimalK = detectElbowPoint(kValues, sseValues);
            
            // Get best model results
            const bestModelResult = resultsByK[optimalK];
            const optimalCentroids = bestModelResult.centroids;

            // Render quantized image & details
            renderQuantizedImage(optimalCentroids);
            renderColorPalette(optimalCentroids);
            renderElbowChart(kValues, sseValues, optimalK);

            // Update UI elements
            document.getElementById('quantized-k-label').textContent = `K = ${optimalK} Colores`;
            document.getElementById('opt-k-val').textContent = optimalK;

        } catch (err) {
            console.error("Error running K-Means analysis:", err);
            alert("Ocurrió un error en la computación. Intente de nuevo.");
        } finally {
            // Hide loading spinner
            loader.classList.add('opacity-0', 'pointer-events-none');
        }
    }, 100);
}

// Render the quantized image using optimal K on target canvas
function renderQuantizedImage(centroids) {
    const canvasQuant = document.getElementById('canvas-quantized');
    const ctxQuant = canvasQuant.getContext('2d');
    
    // We render at 300x300 size for smooth and fast client-side performance
    canvasQuant.width = 300;
    canvasQuant.height = 300;
    
    // Draw original image downsampled to 300x300
    ctxQuant.drawImage(currentImageData, 0, 0, 300, 300);
    const imgDataObj = ctxQuant.getImageData(0, 0, 300, 300);
    const data = imgDataObj.data;
    
    // Instantiate a virtual classifier with converged centroids
    const classifier = new KMeans(centroids.length);
    classifier.centroids = centroids;

    // Map each of the 300x300 pixels to its nearest centroid
    for (let i = 0; i < data.length; i += 4) {
        const pixel = [data[i], data[i+1], data[i+2]];
        const closestCentroid = classifier.predict(pixel);
        
        data[i]   = Math.round(closestCentroid[0]);
        data[i+1] = Math.round(closestCentroid[1]);
        data[i+2] = Math.round(closestCentroid[2]);
    }
    
    ctxQuant.putImageData(imgDataObj, 0, 0);
}

// Display the color blocks in the palette panel
function renderColorPalette(centroids) {
    const container = document.getElementById('palette-container');
    container.innerHTML = ''; // Clear prior entries
    
    // Sort centroids by luminance/brightness to make palette orderly
    const sortedCentroids = [...centroids].sort((a, b) => {
        const lumA = 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2];
        const lumB = 0.299 * b[0] + 0.587 * b[1] + 0.114 * b[2];
        return lumB - lumA; // Darker to lighter or lighter to darker
    });

    sortedCentroids.forEach((centroid, index) => {
        const r = Math.round(centroid[0]);
        const g = Math.round(centroid[1]);
        const b = Math.round(centroid[2]);
        const hex = rgbToHex(r, g, b);
        
        // Calculate text color for readable labels inside blocks
        const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
        const textColorClass = luminance > 0.5 ? 'text-gray-900' : 'text-white';
        
        const block = document.createElement('div');
        block.className = 'palette-block flex-1 flex flex-col justify-end p-4 h-36 border border-gray-800/40 relative';
        block.style.backgroundColor = hex;
        block.onclick = () => copyToClipboard(hex);
        
        block.innerHTML = `
            <div class="${textColorClass} font-bold text-sm tracking-wide bg-black/15 p-1 rounded backdrop-blur-2xs w-fit">
                ${hex}
            </div>
            <div class="${textColorClass} text-2xs opacity-80 mt-1">
                Color ${index + 1}
            </div>
        `;
        
        container.appendChild(block);
    });
}

// Convert RGB components to HEX string
function rgbToHex(r, g, b) {
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase();
}

// Helper to copy Hex codes to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        const toast = document.getElementById('copy-toast');
        toast.classList.remove('opacity-0', 'translate-y-2', 'pointer-events-none');
        
        setTimeout(() => {
            toast.classList.add('opacity-0', 'translate-y-2', 'pointer-events-none');
        }, 2000);
    });
}

// Render the Line Chart using Chart.js
function renderElbowChart(kValues, sseValues, optimalK) {
    const ctx = document.getElementById('chart-elbow').getContext('2d');
    
    // Destroy previous instance to avoid rendering overlaps
    if (chartInstance) {
        chartInstance.destroy();
    }
    
    // Highlight colors
    const datasetColor = '#b86dc1';
    const highlightColor = '#ff8c00';

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: kValues.map(k => `K=${k}`),
            datasets: [{
                label: 'Inercia Intra-Clúster (SSE)',
                data: sseValues,
                borderColor: datasetColor,
                backgroundColor: 'rgba(184, 109, 193, 0.15)',
                borderWidth: 3,
                pointBackgroundColor: kValues.map(k => k === optimalK ? highlightColor : datasetColor),
                pointBorderColor: kValues.map(k => k === optimalK ? '#fff' : datasetColor),
                pointRadius: kValues.map(k => k === optimalK ? 8 : 5),
                pointHoverRadius: 9,
                fill: true,
                tension: 0.35
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return ` SSE: ${context.raw.toFixed(1)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.6)'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.6)'
                    },
                    title: {
                        display: true,
                        text: 'Suma de Errores Cuadráticos (Inercia)',
                        color: 'rgba(255, 255, 255, 0.4)',
                        font: {
                            size: 10
                        }
                    }
                }
            }
        }
    });
}
