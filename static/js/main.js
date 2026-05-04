document.addEventListener('DOMContentLoaded', () => {
    const $ = (id) => document.getElementById(id);

    if (window.lucide) {
        window.lucide.createIcons();
    }

    function updateHeaderClock() {
        const headerTime = $('headerTime');
        if (!headerTime) return;
        const now = new Date();
        headerTime.innerText = new Intl.DateTimeFormat(undefined, {
            month: 'short',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        }).format(now);
    }

    updateHeaderClock();
    window.setInterval(updateHeaderClock, 30000);

    const sidebarLinks = Array.from(document.querySelectorAll('.sidebar-link[data-section]'));
    const sectionTargets = sidebarLinks
        .map((link) => document.getElementById(link.dataset.section))
        .filter(Boolean);

    function setActiveSidebar(sectionId) {
        sidebarLinks.forEach((link) => {
            link.classList.toggle('active', link.dataset.section === sectionId);
        });
    }

    let manualActiveUntil = 0;
    let scrollSpyQueued = false;

    function updateActiveFromScroll() {
        scrollSpyQueued = false;
        if (Date.now() < manualActiveUntil || !sectionTargets.length) return;

        const referenceY = Math.min(170, Math.max(110, window.innerHeight * 0.22));
        let bestSection = null;
        let bestScore = Number.POSITIVE_INFINITY;

        sectionTargets.forEach((section) => {
            const rect = section.getBoundingClientRect();
            if (rect.bottom < referenceY || rect.top > window.innerHeight) return;
            const score = Math.abs(rect.top - referenceY);
            if (score < bestScore) {
                bestScore = score;
                bestSection = section;
            }
        });

        if (bestSection) {
            setActiveSidebar(bestSection.id);
        }
    }

    function queueScrollSpy() {
        if (scrollSpyQueued) return;
        scrollSpyQueued = true;
        window.requestAnimationFrame(updateActiveFromScroll);
    }

    sidebarLinks.forEach((link) => {
        link.addEventListener('click', (event) => {
            const target = document.getElementById(link.dataset.section);
            if (!target) return;
            event.preventDefault();
            manualActiveUntil = Date.now() + 1200;
            setActiveSidebar(link.dataset.section);
            target.scrollIntoView({ behavior: 'smooth', block: 'start', inline: 'nearest' });
            if (window.history && window.history.replaceState) {
                window.history.replaceState(null, '', link.getAttribute('href'));
            }
        });
    });

    const initialHash = window.location.hash ? window.location.hash.slice(1) : 'dashboard-section';
    if (document.getElementById(initialHash)) {
        setActiveSidebar(initialHash);
    }
    window.addEventListener('scroll', queueScrollSpy, { passive: true });
    window.addEventListener('resize', queueScrollSpy);
    const statusPanel = document.querySelector('.status-panel');
    if (statusPanel) {
        statusPanel.addEventListener('scroll', queueScrollSpy, { passive: true });
    }
    window.setTimeout(queueScrollSpy, 250);

    window.toggleDetails = function (id) {
        const el = $(id);
        if (el) {
            el.classList.toggle('show');
        }
    };

    const fileInput = $('fileInput');
    const badge = $('fileTypeBadge');
    const uploadForm = $('uploadForm');
    const predictBtn = $('predictBtn');
    const dropZone = $('dropZone');
    const uploadFileName = $('uploadFileName');
    const emergencyModal = $('emergencyModal');
    const emergencyModalMessage = $('emergencyModalMessage');
    const emergencyModalWindow = $('emergencyModalWindow');
    const emergencyCloseBtn = $('emergencyCloseBtn');
    const emergencyViewDetailsBtn = $('emergencyViewDetailsBtn');
    const sampleButtons = document.querySelectorAll('.sample-btn');

    let map = null;
    let lastSourceLabel = 'awaiting upload';
    let lastFileType = '--';

    function setText(id, value) {
        const el = $(id);
        if (el) {
            el.innerText = value;
        }
    }

    function formatNumber(value, digits = 2) {
        const num = Number(value);
        if (!Number.isFinite(num)) return 'N/A';
        return num.toFixed(digits);
    }

    function formatLoose(value) {
        if (value === undefined || value === null || value === '') return 'N/A';
        if (typeof value === 'number') return Number.isInteger(value) ? String(value) : String(value);
        return String(value);
    }

    function clearError() {
        const errorBanner = $('errorBanner');
        if (errorBanner) {
            errorBanner.style.display = 'none';
            errorBanner.innerText = '';
        }
    }

    function showError(message) {
        const errorBanner = $('errorBanner');
        if (errorBanner) {
            errorBanner.style.display = 'block';
            errorBanner.innerText = message || 'Unknown error';
        } else {
            alert(message || 'Unknown error');
        }
    }

    function setPredictBusy(isBusy) {
        if (!predictBtn) return;
        if (!predictBtn.dataset.defaultHtml) {
            predictBtn.dataset.defaultHtml = predictBtn.innerHTML;
        }
        if (isBusy) {
            predictBtn.innerHTML = '<span class="spinner-border spinner-border-sm" aria-hidden="true"></span><span>Processing</span>';
            predictBtn.disabled = true;
        } else {
            predictBtn.innerHTML = predictBtn.dataset.defaultHtml;
            predictBtn.disabled = false;
            if (window.lucide) window.lucide.createIcons();
        }
    }

    function setSamplesBusy(isBusy) {
        sampleButtons.forEach((button) => {
            button.disabled = isBusy;
        });
    }

    function emergencyMessage(seconds) {
        const value = Number.isFinite(Number(seconds)) ? Number(seconds).toFixed(2) : 'X';
        return `Strong earthquake shaking may begin in approximately ${value} seconds.\nTake immediate action: Move to a safe place. Drop, Cover, and Hold On.`;
    }

    function closeEmergencyModal() {
        if (emergencyModal) {
            emergencyModal.style.display = 'none';
        }
    }

    function openEmergencyModal(message, warningWindowSec) {
        if (!emergencyModal || !emergencyModalMessage || !emergencyModalWindow) return;
        emergencyModalMessage.innerText = message;
        emergencyModalWindow.innerText = `${formatNumber(warningWindowSec, 2)} seconds`;
        emergencyModal.style.display = 'flex';
    }

    if (emergencyCloseBtn) {
        emergencyCloseBtn.addEventListener('click', closeEmergencyModal);
    }

    if (emergencyViewDetailsBtn) {
        emergencyViewDetailsBtn.addEventListener('click', () => {
            closeEmergencyModal();
            const target = $('resultsContainer');
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    }

    function updateSourceLine(data) {
        const type = data && data.file_type ? data.file_type : lastFileType;
        const filename = data && data.filename ? data.filename : lastSourceLabel;
        lastSourceLabel = filename;
        if (uploadFileName && filename && filename !== 'awaiting upload') {
            uploadFileName.innerText = filename;
        }
        setText('sourceMeta', `Source: ${filename} | Type: ${type}`);
        setText('stationMeta', 'Station: Demo STN-01');
    }

    function updateFileBadge(file) {
        if (!badge) return;

        if (!file) {
            badge.className = 'file-badge';
            badge.innerText = 'No File';
            lastFileType = '--';
            return;
        }

        const ext = file.name.split('.').pop().toUpperCase();
        lastFileType = ext;
        badge.className = 'file-badge selected';
        badge.innerText = ext;
        if (uploadFileName) {
            uploadFileName.innerText = file.name;
        }
        lastSourceLabel = file.name;
        updateSourceLine({ file_type: ext });
    }

    function isSupportedFile(file) {
        return file && file.name.toLowerCase().endsWith('.npy');
    }

    if (fileInput && badge) {
        fileInput.addEventListener('change', function () {
            clearError();
            const file = this.files && this.files.length > 0 ? this.files[0] : null;
            if (file && !isSupportedFile(file)) {
                showError('Unsupported file. Please upload a .npy waveform.');
                this.value = '';
                updateFileBadge(null);
                if (uploadFileName) uploadFileName.innerText = 'Select a 3-channel .npy sample';
                return;
            }
            updateFileBadge(file);
        });
    }

    if (dropZone && fileInput) {
        ['dragenter', 'dragover'].forEach((eventName) => {
            dropZone.addEventListener(eventName, (event) => {
                event.preventDefault();
                event.stopPropagation();
                dropZone.classList.add('is-dragging');
            });
        });

        ['dragleave', 'drop'].forEach((eventName) => {
            dropZone.addEventListener(eventName, (event) => {
                event.preventDefault();
                event.stopPropagation();
                dropZone.classList.remove('is-dragging');
            });
        });

        dropZone.addEventListener('drop', (event) => {
            const file = event.dataTransfer && event.dataTransfer.files.length
                ? event.dataTransfer.files[0]
                : null;
            if (!file) return;
            if (!isSupportedFile(file)) {
                showError('Unsupported file. Please upload a .npy waveform.');
                return;
            }
            const transfer = new DataTransfer();
            transfer.items.add(file);
            fileInput.files = transfer.files;
            updateFileBadge(file);
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (event) => {
            event.preventDefault();

            if (!fileInput || !fileInput.files.length) {
                showError('Select a .npy waveform before prediction.');
                return;
            }

            clearError();
            closeEmergencyModal();
            setPredictBusy(true);

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
                showError('An error occurred during prediction.');
            } finally {
                setPredictBusy(false);
            }
        });
    }

    window.loadSample = async function (sampleType) {
        clearError();
        closeEmergencyModal();
        setSamplesBusy(true);

        const plotPlaceholder = $('plotPlaceholderText');
        const plotDiv = $('waveform-plot');
        if (plotPlaceholder && plotDiv) {
            plotPlaceholder.style.display = 'flex';
            plotPlaceholder.innerText = 'Loading sample waveform...';
            plotDiv.style.display = 'none';
        }

        lastSourceLabel = sampleType === 'earthquake' ? 'earthquake_demo.npy' : 'noise_demo.npy';
        lastFileType = 'NPY (Sample)';
        if (badge) {
            badge.className = 'file-badge selected';
            badge.innerText = 'Sample';
        }
        updateSourceLine({ file_type: lastFileType });

        try {
            const response = await fetch(`/sample/${sampleType}`);
            const data = await response.json();
            renderResults(data);
        } catch (err) {
            console.error(err);
            showError('Failed to load sample.');
        } finally {
            setSamplesBusy(false);
            if (window.lucide) window.lucide.createIcons();
        }
    };

    function getPredictionState(prediction) {
        const normalized = (prediction || '').toLowerCase();
        if (normalized === 'earthquake') {
            return {
                key: 'warning',
                alertText: 'WARNING',
                className: 'text-red'
            };
        }
        if (normalized.includes('possible')) {
            return {
                key: 'possible',
                alertText: 'POSSIBLE EVENT',
                className: 'text-warning'
            };
        }
        if (normalized === 'noise') {
            return {
                key: 'safe',
                alertText: 'SAFE',
                className: 'text-green'
            };
        }
        return {
            key: 'neutral',
            alertText: 'STANDBY',
            className: ''
        };
    }

    function formatDecisionReason(reason) {
        const labels = {
            high_confidence: 'High confidence event',
            below_threshold: 'Below detection threshold',
            low_energy_filtered: 'Low signal energy filtered',
            possible_event_low_confidence: 'Possible event, low confidence',
            invalid_phase: 'Invalid P/S phase timing'
        };
        return labels[reason] || formatLoose(reason);
    }

    function updateStatusCards(data, pSec, sSec) {
        const prediction = data.prediction || 'Unknown';
        const state = getPredictionState(prediction);
        const cardClass = $('cardClass');
        const resClassText = $('resClassText');
        const resAlertStatus = $('resAlertStatus');
        const predictionBadge = $('predictionBadge');

        if (resClassText) {
            resClassText.innerText = prediction;
            resClassText.className = state.className;
        }

        if (cardClass) {
            cardClass.classList.remove('state-safe', 'state-warning', 'state-possible');
            if (state.key !== 'neutral') {
                cardClass.classList.add(`state-${state.key}`);
            }
        }

        if (resAlertStatus) {
            resAlertStatus.innerText = state.alertText;
            resAlertStatus.className = `alert-chip ${state.key === 'neutral' ? 'standby' : state.key}`;
        }

        if (predictionBadge) {
            predictionBadge.innerText = prediction;
            predictionBadge.className = `prediction-badge ${state.key}`;
        }

        setText('resConfidence', `${formatNumber(data.confidence, 2)}%`);
        setText('resDecisionReason', formatDecisionReason(data.decision_reason));
        setText('resThreshold', formatLoose(data.threshold_used));
        setText('resAlert', data.alert === true ? 'true' : 'false');
        setText('resEnergy', formatLoose(data.signal_energy));
        setText('resVariance', formatLoose(data.signal_variance));

        if (state.key === 'warning' && data.alert) {
            setText('resExplanation', 'High-confidence seismic event detected. Alert protocols are active.');
        } else if (state.key === 'possible') {
            setText('resExplanation', 'Signal features suggest a possible seismic event, but confidence is below the confirmed event threshold.');
        } else if (state.key === 'safe') {
            setText('resExplanation', 'No strong seismic event detected in the processed waveform.');
        } else {
            setText('resExplanation', 'Awaiting a waveform or demo sample.');
        }

        if (data.phase_status === 'invalid') {
            setText('resPhaseNote', 'P/S wave timing is not reliable for this sample.');
        } else if (data.phase_status === 'missing') {
            setText('resPhaseNote', 'P/S timing unavailable for this sample.');
        } else {
            setText('resPhaseNote', '');
        }

        setText('resPWave', formatArrival(pSec));
        setText('resSWave', formatArrival(sSec));

        const spGap = data.s_p_gap_sec;
        if (spGap !== undefined && spGap !== null) {
            setText('resPSGap', `${formatNumber(spGap, 2)} sec`);
        } else if (Number.isFinite(Number(sSec)) && Number.isFinite(Number(pSec))) {
            setText('resPSGap', `${formatNumber(Number(sSec) - Number(pSec), 2)} sec`);
        } else {
            setText('resPSGap', 'N/A');
        }
    }

    function formatArrival(sec) {
        const value = Number(sec);
        if (!Number.isFinite(value)) return 'N/A';
        return `${formatNumber(value, 2)} sec (${Math.round(value * 100)} samples)`;
    }

    function updateEmergency(data) {
        const cardEmergency = $('cardEmergency');
        const warningWindow = Number(data.warning_time_sec);
        const shouldShowEmergency = (
            data.prediction === 'Earthquake' &&
            data.alert === true &&
            Number.isFinite(warningWindow) &&
            warningWindow > 0
        );

        if (!cardEmergency) return;

        if (shouldShowEmergency) {
            const message = emergencyMessage(warningWindow);
            cardEmergency.style.display = 'block';
            setText('resEmergencyMessage', message);
            setText('resWarningWindow', `${formatNumber(warningWindow, 2)} seconds`);
            openEmergencyModal(message, warningWindow);
        } else {
            cardEmergency.style.display = 'none';
            closeEmergencyModal();
        }
    }

    function updateMagnitude(data) {
        const mag = Number(data.magnitude);
        const resMagLabel = $('resMagLabel');
        const magBar = $('resMagBar');

        if (!Number.isFinite(mag)) {
            setText('resMagVal', 'N/A');
            if (resMagLabel) {
                resMagLabel.innerText = 'N/A';
                resMagLabel.style.backgroundColor = '';
                resMagLabel.style.color = '';
            }
            if (magBar) {
                magBar.style.width = '0%';
                magBar.style.backgroundColor = 'var(--text-muted)';
            }
            return;
        }

        const magScale = getMagnitudeScale(mag);
        setText('resMagVal', mag.toFixed(1));
        if (resMagLabel) {
            resMagLabel.innerText = magScale.label;
            resMagLabel.style.backgroundColor = magScale.color;
            resMagLabel.style.color = magScale.textColor;
            resMagLabel.style.borderColor = 'transparent';
        }
        if (magBar) {
            magBar.style.width = `${Math.min((mag / 10) * 100, 100)}%`;
            magBar.style.backgroundColor = magScale.color;
        }
    }

    function drawWaveform(data, pSec, sSec) {
        const plotPlaceholder = $('plotPlaceholderText');
        const plotDiv = $('waveform-plot');
        const liveBadge = $('liveBadge');

        if (!plotPlaceholder || !plotDiv) return;

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

        if (!x || !e || !n || !z || !window.Plotly) {
            plotPlaceholder.style.display = 'flex';
            plotPlaceholder.innerText = window.Plotly ? 'Waveform data unavailable' : 'Plotly unavailable';
            plotDiv.style.display = 'none';
            if (liveBadge) liveBadge.style.display = 'none';
            return;
        }

        plotPlaceholder.style.display = 'none';
        plotDiv.style.display = 'block';
        if (liveBadge) liveBadge.style.display = 'inline-flex';

        const traces = [
            {
                x,
                y: e,
                xaxis: 'x',
                yaxis: 'y',
                mode: 'lines',
                name: 'E Channel',
                line: { color: '#22d3ee', width: 1.25 }
            },
            {
                x,
                y: n,
                xaxis: 'x2',
                yaxis: 'y2',
                mode: 'lines',
                name: 'N Channel',
                line: { color: '#3b82f6', width: 1.25 }
            },
            {
                x,
                y: z,
                xaxis: 'x3',
                yaxis: 'y3',
                mode: 'lines',
                name: 'Z Channel',
                line: { color: '#f97316', width: 1.25 }
            }
        ];

        const shapes = [];
        const annotations = [];
        addPhaseMarker(shapes, annotations, pSec, 'P-wave', '#22d3ee');
        addPhaseMarker(shapes, annotations, sSec, 'S-wave', '#ef4444');

        const yAxisTitle = data.waveform_normalized ? 'Normalized Amplitude' : 'Amplitude';
        const axisBase = {
            showgrid: true,
            gridcolor: 'rgba(135, 162, 194, 0.12)',
            zeroline: false,
            color: '#8ea4bd',
            tickfont: { size: 10 },
            linecolor: 'rgba(135, 162, 194, 0.16)'
        };

        const yAxisBase = {
            ...axisBase,
            range: data.waveform_normalized ? [-1.15, 1.15] : undefined,
            title: { text: yAxisTitle, font: { size: 10 } }
        };

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: '#07111f',
            font: { color: '#edf6ff', family: 'Inter, sans-serif' },
            margin: { t: 30, b: 45, l: 54, r: 20 },
            grid: { rows: 3, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
            xaxis: { ...axisBase, showticklabels: false },
            xaxis2: { ...axisBase, showticklabels: false },
            xaxis3: { ...axisBase, title: { text: 'Samples', font: { size: 11 } } },
            yaxis: { ...yAxisBase, title: { text: 'E', font: { size: 11 } } },
            yaxis2: { ...yAxisBase, title: { text: 'N', font: { size: 11 } } },
            yaxis3: { ...yAxisBase, title: { text: 'Z', font: { size: 11 } } },
            shapes,
            annotations,
            showlegend: true,
            legend: {
                orientation: 'h',
                y: -0.15,
                x: 0,
                font: { size: 11 },
                bgcolor: 'rgba(0,0,0,0)'
            }
        };

        Plotly.newPlot(plotDiv, traces, layout, {
            responsive: true,
            displayModeBar: false
        });
    }

    function addPhaseMarker(shapes, annotations, seconds, label, color) {
        const sec = Number(seconds);
        if (!Number.isFinite(sec)) return;

        const sample = sec * 100;
        [
            { xref: 'x', yref: 'y domain' },
            { xref: 'x2', yref: 'y2 domain' },
            { xref: 'x3', yref: 'y3 domain' }
        ].forEach((axis) => {
            shapes.push({
                type: 'line',
                xref: axis.xref,
                yref: axis.yref,
                x0: sample,
                x1: sample,
                y0: 0,
                y1: 1,
                line: { color, dash: 'dot', width: 1.6 }
            });
        });

        annotations.push({
            x: sample,
            y: 1.02,
            xref: 'x',
            yref: 'paper',
            text: label,
            showarrow: false,
            font: { color, size: 11 },
            bgcolor: 'rgba(5, 9, 20, 0.84)',
            bordercolor: color,
            borderpad: 3
        });
    }

    function updateLocationAndMap(data) {
        const lat = data.latitude;
        const lon = data.longitude;
        const depth = data.depth;
        const hasLocation = lat !== undefined && lat !== null && lon !== undefined && lon !== null;
        const cardMap = $('cardMap');
        const mapContainer = $('mapContainer');
        const mapPlaceholder = $('mapPlaceholder');

        setText('resLocStatus', data.location_status || 'experimental');

        if (!hasLocation) {
            setText('resLat', 'N/A');
            setText('resLon', 'N/A');
            setText('resDepth', 'N/A');
            if (map) {
                map.remove();
                map = null;
            }
            if (cardMap) cardMap.style.display = 'block';
            if (mapContainer) mapContainer.style.display = 'none';
            if (mapPlaceholder) {
                mapPlaceholder.style.display = 'flex';
                mapPlaceholder.innerHTML = '<i data-lucide="map-pinned"></i><span>Location experimental</span>';
            }
            if (window.lucide) window.lucide.createIcons();
            return;
        }

        setText('resLat', parseFloat(lat).toFixed(4));
        setText('resLon', parseFloat(lon).toFixed(4));
        setText('resDepth', depth !== undefined && depth !== null ? parseFloat(depth).toFixed(1) : 'N/A');

        if (!window.L || !mapContainer) {
            if (mapPlaceholder) {
                mapPlaceholder.style.display = 'flex';
                mapPlaceholder.innerText = 'Location experimental';
            }
            return;
        }

        if (cardMap) cardMap.style.display = 'block';
        if (mapPlaceholder) mapPlaceholder.style.display = 'none';
        mapContainer.style.display = 'block';

        if (map) {
            map.remove();
            map = null;
        }

        map = L.map('mapContainer').setView([lat, lon], 6);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
        }).addTo(map);

        const marker = L.marker([lat, lon]).addTo(map);
        marker.bindPopup('<b>Predicted earthquake location</b>').openPopup();

        setTimeout(() => {
            if (map) map.invalidateSize();
        }, 250);
    }

    function renderResults(data) {
        if (!data || data.status !== 'success') {
            showError((data && data.message) || 'Prediction failed.');
            return;
        }

        clearError();

        const resultsContainer = $('resultsContainer');
        if (resultsContainer) {
            resultsContainer.style.display = 'grid';
            resultsContainer.classList.add('has-results');
        }

        const pSec = data.p_arrival_sec ?? data.p_arrival;
        const sSec = data.s_arrival_sec ?? data.s_arrival;

        updateSourceLine(data);
        updateStatusCards(data, pSec, sSec);
        updateEmergency(data);
        updateMagnitude(data);
        drawWaveform(data, pSec, sSec);
        updateLocationAndMap(data);
    }

    function getMagnitudeScale(mag) {
        if (mag < 2.0) return { label: 'Minor', color: '#22d3ee', textColor: '#001018' };
        if (mag < 4.0) return { label: 'Light', color: '#3b82f6', textColor: '#f8fbff' };
        if (mag < 6.0) return { label: 'Moderate', color: '#f59e0b', textColor: '#241100' };
        return { label: 'Strong', color: '#ef4444', textColor: '#210606' };
    }
});
