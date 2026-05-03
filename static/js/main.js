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
    const emergencyModal = document.getElementById('emergencyModal');
    const emergencyModalMessage = document.getElementById('emergencyModalMessage');
    const emergencyModalWindow = document.getElementById('emergencyModalWindow');
    const emergencyCloseBtn = document.getElementById('emergencyCloseBtn');
    const emergencyViewDetailsBtn = document.getElementById('emergencyViewDetailsBtn');

    function closeEmergencyModal() {
        if (emergencyModal) {
            emergencyModal.style.display = 'none';
        }
    }

    function openEmergencyModal(message, warningWindowSec) {
        if (!emergencyModal) return;
        emergencyModalMessage.innerText = message;
        emergencyModalWindow.innerText = `${warningWindowSec.toFixed(2)} seconds`;
        emergencyModal.style.display = 'flex';
    }

    if (emergencyCloseBtn) {
        emergencyCloseBtn.addEventListener('click', closeEmergencyModal);
    }

    if (emergencyViewDetailsBtn) {
        emergencyViewDetailsBtn.addEventListener('click', () => {
            closeEmergencyModal();
            const target = document.getElementById('resultsContainer');
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    }

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

            const errorBanner = document.getElementById('errorBanner');
            if (errorBanner) {
                errorBanner.style.display = 'none';
                errorBanner.innerText = '';
            }
            closeEmergencyModal();

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
        const errorBanner = document.getElementById('errorBanner');
        if (errorBanner) {
            errorBanner.style.display = 'none';
            errorBanner.innerText = '';
        }
        closeEmergencyModal();
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
        const errorBanner = document.getElementById('errorBanner');
        if (data.status !== "success") {
            if (errorBanner) {
                errorBanner.style.display = 'block';
                errorBanner.innerText = data.message || "Unknown error";
            } else {
                alert("Prediction failed: " + (data.message || "Unknown error"));
            }
            const resultsContainer = document.getElementById('resultsContainer');
            if (resultsContainer) {
                resultsContainer.style.display = 'none';
            }
            return;
        }
        if (errorBanner) {
            errorBanner.style.display = 'none';
            errorBanner.innerText = '';
        }

        const resultsContainer = document.getElementById('resultsContainer');
        resultsContainer.style.display = 'block';

        // 1. Classification
        const cardClass = document.getElementById('cardClass');
        const resClassText = document.getElementById('resClassText');
        const resConfidence = document.getElementById('resConfidence');
        const resAlertStatus = document.getElementById('resAlertStatus');
        const resDecisionReason = document.getElementById('resDecisionReason');
        const resExplanation = document.getElementById('resExplanation');
        const resPhaseNote = document.getElementById('resPhaseNote');
        const cardEmergency = document.getElementById('cardEmergency');
        const resEmergencyMessage = document.getElementById('resEmergencyMessage');
        const resWarningWindow = document.getElementById('resWarningWindow');

        resClassText.innerText = data.prediction;
        resConfidence.innerText = data.confidence + '% Confidence';

        const predLower = (data.prediction || '').toLowerCase();
        if (predLower === 'earthquake') {
            resClassText.className = 'mb-0 text-red fw-bold';
            cardClass.style.borderColor = '#ff4444';
            resAlertStatus.innerText = 'WARNING';
            resAlertStatus.className = 'badge alert-badge alert-warning';
        } else if (predLower.includes('possible')) {
            resClassText.className = 'mb-0 text-warning fw-bold';
            cardClass.style.borderColor = '#ffc107';
            resAlertStatus.innerText = 'POSSIBLE EVENT';
            resAlertStatus.className = 'badge alert-badge alert-possible';
        } else {
            resClassText.className = 'mb-0 text-cyan fw-bold';
            cardClass.style.borderColor = '#00e5ff';
            resAlertStatus.innerText = 'SAFE';
            resAlertStatus.className = 'badge alert-badge alert-safe';
        }

        resDecisionReason.innerText = data.decision_reason || 'N/A';

        if (data.early_warning) {
            cardEmergency.style.display = 'block';
            resEmergencyMessage.innerText = data.emergency_message || 'P-wave detected.';
            resWarningWindow.innerText = `${data.warning_time_sec ?? 'N/A'} seconds`;
        } else {
            cardEmergency.style.display = 'none';
        }

        const shouldShowEmergencyModal = (
            data.prediction === 'Earthquake' &&
            data.alert === true &&
            data.early_warning === true &&
            typeof data.warning_time_sec === 'number' &&
            data.warning_time_sec > 0
        );

        if (shouldShowEmergencyModal) {
            const modalMessage = data.emergency_message ||
                `Strong earthquake shaking may begin in approximately ${data.warning_time_sec.toFixed(2)} seconds.\n\nTake immediate action:\n• Move to a safe place\n• Drop, Cover, and Hold On\n\nEstimated time remaining: ${data.warning_time_sec.toFixed(2)} seconds`;
            openEmergencyModal(modalMessage, data.warning_time_sec);
        } else {
            closeEmergencyModal();
        }

        // Decision details
        document.getElementById('resThreshold').innerText = data.threshold_used ?? 'N/A';
        document.getElementById('resAlert').innerText = data.alert ? 'true' : 'false';
        document.getElementById('resEnergy').innerText = data.signal_energy ?? 'N/A';
        document.getElementById('resVariance').innerText = data.signal_variance ?? 'N/A';

        if (predLower === 'earthquake' && data.alert) {
            resExplanation.innerText = 'High-confidence seismic event detected.';
        } else if (predLower.includes('possible')) {
            resExplanation.innerText = 'Signal is suspicious but confidence is not high enough for a confirmed alert.';
        } else {
            resExplanation.innerText = 'No strong seismic event detected.';
        }

        if (data.phase_status === 'invalid') {
            resPhaseNote.innerText = 'P/S wave timing is not reliable for this sample.';
        } else {
            resPhaseNote.innerText = '';
        }

        // 2. Phase Picking
        const p_sec = data.p_arrival_sec ?? data.p_arrival;
        const s_sec = data.s_arrival_sec ?? data.s_arrival;
        const sp_gap = data.s_p_gap_sec;

        document.getElementById('resPWave').innerText = p_sec ? `${Math.round(p_sec * 100)} samples (${p_sec} sec)` : 'N/A';
        document.getElementById('resSWave').innerText = s_sec ? `${Math.round(s_sec * 100)} samples (${s_sec} sec)` : 'N/A';
        if (sp_gap !== undefined && sp_gap !== null) {
            document.getElementById('resPSGap').innerText = `${sp_gap} sec`;
        } else if (s_sec && p_sec) {
            document.getElementById('resPSGap').innerText = `${(s_sec - p_sec).toFixed(2)} sec`;
        } else {
            document.getElementById('resPSGap').innerText = 'N/A';
        }

        // 3. Magnitude
        const mag = data.magnitude;
        const resMagVal = document.getElementById('resMagVal');
        const resMagLabel = document.getElementById('resMagLabel');
        const magBar = document.getElementById('resMagBar');
        if (mag === undefined || mag === null) {
            resMagVal.innerText = 'N/A';
            resMagLabel.innerText = 'N/A';
            resMagLabel.style.backgroundColor = '#6c757d';
            resMagLabel.style.color = '#fff';
            magBar.style.width = '0%';
            magBar.style.backgroundColor = '#6c757d';
        } else {
            resMagVal.innerText = mag.toFixed(1);
            const magScale = getMagnitudeScale(mag);
            resMagLabel.innerText = magScale.label;
            resMagLabel.style.backgroundColor = magScale.color;
            resMagLabel.style.color = '#000';
            magBar.style.width = Math.min((mag / 10) * 100, 100) + '%';
            magBar.style.backgroundColor = magScale.color;
        }

        // 4. Update plot label
        const plotTitle = document.getElementById('plotTitle');
        plotTitle.innerText = `Prediction: ${data.prediction} (${data.confidence}%)`;
        document.getElementById('liveBadge').style.display = 'block';

        // Waveform plot
        const plotPlaceholder = document.getElementById('plotPlaceholderText');
        const plotDiv = document.getElementById('waveform-plot');
        plotDiv.innerHTML = '';

        let x = data.waveform_x;
        let e = data.waveform_e;
        let n = data.waveform_n;
        let z = data.waveform_z;

        if ((!x || !e || !n || !z) && Array.isArray(data.waveform) && data.waveform.length === 3) {
            e = data.waveform[0];
            n = data.waveform[1];
            z = data.waveform[2];
            x = Array.from({ length: e.length }, (_, i) => i);
        }

        if (!x || !e || !n || !z) {
            plotPlaceholder.style.display = 'block';
            plotPlaceholder.innerText = 'Waveform data unavailable';
            plotDiv.style.display = 'none';
            return;
        }

        plotPlaceholder.style.display = 'none';
        plotDiv.style.display = 'block';

        const traceE = { x: x, y: e, mode: 'lines', name: 'East', line: { color: '#00e5ff', width: 1 } };
        const traceN = { x: x, y: n, mode: 'lines', name: 'North', line: { color: '#ff4444', width: 1 } };
        const traceZ = { x: x, y: z, mode: 'lines', name: 'Vertical', line: { color: '#a55eea', width: 1 } };

        // P/S lines using shapes
        const shapes = [];
        if (p_sec) {
            shapes.push({ type: 'line', x0: p_sec * 100, x1: p_sec * 100, y0: 0, y1: 1, yref: 'paper', line: { color: '#00e5ff', dash: 'dash', width: 2 } });
        }
        if (s_sec) {
            shapes.push({ type: 'line', x0: s_sec * 100, x1: s_sec * 100, y0: 0, y1: 1, yref: 'paper', line: { color: '#ff4444', dash: 'dash', width: 2 } });
        }

        const yAxisTitle = data.waveform_normalized ? 'Normalized Amplitude' : 'Amplitude';
        const layout = {
            paper_bgcolor: '#1a1d2e',
            plot_bgcolor: '#1a1d2e',
            font: { color: '#f8f9fa' },
            margin: { t: 30, b: 40, l: 40, r: 20 },
            xaxis: { title: 'Samples', gridcolor: '#2a2d3e' },
            yaxis: { title: yAxisTitle, gridcolor: '#2a2d3e' },
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
            document.getElementById('resLocStatus').innerText = data.location_status || 'experimental';

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
