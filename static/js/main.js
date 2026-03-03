document.addEventListener('DOMContentLoaded', () => {

    // Toggle Details logic for Home Page
    window.toggleDetails = function (id) {
        const el = document.getElementById(id);
        if (el) {
            el.classList.toggle('show');
        }
    };

    // Upload Page Elements
    const fileInput = document.getElementById('fileInput');
    const badge = document.getElementById('fileTypeBadge');
    const uploadForm = document.getElementById('uploadForm');
    const predictBtn = document.getElementById('predictBtn');

    if (fileInput && badge) {
        fileInput.addEventListener('change', function () {
            if (this.files && this.files.length > 0) {
                const ext = this.files[0].name.split('.').pop().toUpperCase();
                const colors = { 'NPY': '#00e5ff', 'CSV': '#a55eea', 'HDF5': '#ff4444', 'H5': '#ff4444' };
                badge.style.background = colors[ext] || '#6c757d';
                badge.innerText = ext;
                badge.style.color = '#000';
            } else {
                badge.style.background = '#444';
                badge.innerText = 'No File';
                badge.style.color = '#fff';
            }
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!fileInput.files.length) return;

            const originalText = predictBtn.innerText;
            predictBtn.innerText = 'Processing...';
            predictBtn.disabled = true;

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                renderResults(data);
            } catch (err) {
                console.error(err);
                alert("An error occurred during prediction.");
            } finally {
                predictBtn.innerText = originalText;
                predictBtn.disabled = false;
            }
        });
    }

    window.loadSample = async function (sampleType) {
        const resultsContainer = document.getElementById('resultsContainer');
        if (resultsContainer) {
            resultsContainer.style.display = 'none';
        }
        document.getElementById('plotPlaceholderText').style.display = 'block';
        document.getElementById('waveform-plot').style.display = 'none';

        try {
            const response = await fetch(`/load-sample/${sampleType}`);
            const data = await response.json();
            renderResults(data);
        } catch (err) {
            console.error(err);
            alert("Failed to load sample.");
        }
    }

    let map = null;

    function renderResults(data) {
        if (data.status !== "success") {
            alert("Prediction failed: " + (data.message || "Unknown error"));
            return;
        }

        const resultsContainer = document.getElementById('resultsContainer');
        resultsContainer.style.display = 'block';

        // 1. Classification
        const cardClass = document.getElementById('cardClass');
        const resClassText = document.getElementById('resClassText');
        const resConfidence = document.getElementById('resConfidence');

        resClassText.innerText = data.prediction;
        resConfidence.innerText = data.confidence + '% Confidence';

        if (data.prediction.toLowerCase() === 'earthquake') {
            resClassText.className = 'mb-0 text-red fw-bold';
            cardClass.style.borderColor = '#ff4444';
        } else {
            resClassText.className = 'mb-0 text-cyan fw-bold';
            cardClass.style.borderColor = '#00e5ff';
        }

        // 2. Phase Picking
        const p_sec = data.p_arrival;
        const s_sec = data.s_arrival;

        document.getElementById('resPWave').innerText = p_sec ? `${Math.round(p_sec * 100)} samples (${p_sec} sec)` : 'N/A';
        document.getElementById('resSWave').innerText = s_sec ? `${Math.round(s_sec * 100)} samples (${s_sec} sec)` : 'N/A';
        document.getElementById('resPSGap').innerText = (s_sec && p_sec) ? `${(s_sec - p_sec).toFixed(2)} sec` : 'N/A';

        // 3. Magnitude
        const mag = data.magnitude || 0;
        document.getElementById('resMagVal').innerText = mag.toFixed(1);

        const magScale = getMagnitudeScale(mag);
        const resMagLabel = document.getElementById('resMagLabel');
        resMagLabel.innerText = magScale.label;
        resMagLabel.style.backgroundColor = magScale.color;
        resMagLabel.style.color = '#000';

        const magBar = document.getElementById('resMagBar');
        magBar.style.width = Math.min((mag / 10) * 100, 100) + '%';
        magBar.style.backgroundColor = magScale.color;

        // 4. Update plot label
        const plotTitle = document.getElementById('plotTitle');
        plotTitle.innerText = `Prediction: ${data.prediction} (${data.confidence}%)`;
        document.getElementById('liveBadge').style.display = 'block';

        // Waveform plot
        document.getElementById('plotPlaceholderText').style.display = 'none';
        const plotDiv = document.getElementById('waveform-plot');
        plotDiv.style.display = 'block';
        plotDiv.innerHTML = ''; // before newPlot

        // Assume data.waveform is a 3-element list of arrays of amplitude
        const traceE = { y: data.waveform[0], mode: 'lines', name: 'East', line: { color: '#00e5ff', width: 1 } };
        const traceN = { y: data.waveform[1], mode: 'lines', name: 'North', line: { color: '#ff4444', width: 1 } };
        const traceZ = { y: data.waveform[2], mode: 'lines', name: 'Vertical', line: { color: '#a55eea', width: 1 } };

        // P/S lines using shapes
        const shapes = [];
        if (p_sec) {
            shapes.push({ type: 'line', x0: p_sec * 100, x1: p_sec * 100, y0: 0, y1: 1, yref: 'paper', line: { color: '#00e5ff', dash: 'dash', width: 2 } });
        }
        if (s_sec) {
            shapes.push({ type: 'line', x0: s_sec * 100, x1: s_sec * 100, y0: 0, y1: 1, yref: 'paper', line: { color: '#ff4444', dash: 'dash', width: 2 } });
        }

        const layout = {
            paper_bgcolor: '#1a1d2e',
            plot_bgcolor: '#1a1d2e',
            font: { color: '#f8f9fa' },
            margin: { t: 30, b: 40, l: 40, r: 20 },
            xaxis: { title: 'Samples', gridcolor: '#2a2d3e' },
            yaxis: { title: 'Amplitude', gridcolor: '#2a2d3e' },
            shapes: shapes,
            showlegend: true,
            legend: { orientation: 'h', y: -0.2 }
        };

        Plotly.newPlot(plotDiv, [traceE, traceN, traceZ], layout, { responsive: true });

        // 5. Location and Map
        const lat = data.latitude;
        const lon = data.longitude;
        const depth = data.depth;

        if (lat !== undefined && lon !== undefined && lat !== null && lon !== null) {
            document.getElementById('resLat').innerText = parseFloat(lat).toFixed(4);
            document.getElementById('resLon').innerText = parseFloat(lon).toFixed(4);
            document.getElementById('resDepth').innerText = parseFloat(depth).toFixed(1);

            document.getElementById('cardLoc').style.display = 'block';
            document.getElementById('cardMap').style.display = 'block';

            if (map) { map.remove(); map = null; }

            map = L.map('mapContainer').setView([lat, lon], 6);
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
            }).addTo(map);

            const marker = L.marker([lat, lon]).addTo(map);
            marker.bindPopup("<b>Predicted Earthquake Location</b>").openPopup();

            // Fix map rendering issue in hidden containers
            setTimeout(() => { map.invalidateSize(); }, 200);

        } else {
            document.getElementById('cardLoc').style.display = 'none';
            document.getElementById('cardMap').style.display = 'none';
        }
    }

    function getMagnitudeScale(mag) {
        if (mag < 2.0) return { label: 'Minor', color: '#00e5ff' };
        if (mag < 4.0) return { label: 'Light', color: '#a55eea' };
        if (mag < 6.0) return { label: 'Moderate', color: '#ffa500' };
        return { label: 'Strong', color: '#ff4444' };
    }
});
