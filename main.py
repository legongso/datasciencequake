import sys
import os
import subprocess

required_packages = ['streamlit', 'numpy', 'pandas']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

if os.environ.get('STREAMLIT_RUNNING') != '1':
    os.environ['STREAMLIT_RUNNING'] = '1'
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(stcli.main())

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json

st.set_page_config(
    page_title="세계 지진 군집 분석",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, .stApp, .stApp * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp, [data-testid="stAppViewContainer"] { background: #050608 !important; }
[data-testid="stHeader"], [data-testid="stMain"], [data-testid="stMainBlockContainer"],
.main, .block-container { background: transparent !important; }
.main .block-container {
    padding-top: 1.2rem; padding-bottom: 1rem;
    max-width: 1360px;
}
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}
[data-testid="stAlert"] * { color: rgba(255,255,255,0.85) !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

.hero-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(24px) saturate(140%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.hero-card h1 {
    color: #fff !important;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.03em;
}
.hero-card p {
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.8rem;
    margin: 0.25rem 0 0 0 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-card">
    <h1>세계 지진 군집 분석</h1>
    <p>Global Earthquake Clustering · K-Means (K=3) · 2023+ Dataset</p>
</div>
""", unsafe_allow_html=True)

# ===== 지진 데이터 로드 =====
@st.cache_data
def load_quakes():
    try:
        df = pd.read_csv("quake_points.csv")
    except FileNotFoundError:
        return None, "quake_points.csv 파일을 찾을 수 없습니다."

    # 필수 컬럼 확인 & 정규화
    cols = {c.lower(): c for c in df.columns}
    # 위도/경도 컬럼 자동 탐지
    lat_col = next((cols[k] for k in ['위도', 'lat', 'latitude'] if k in cols), None)
    lon_col = next((cols[k] for k in ['경도', 'lon', 'lng', 'longitude'] if k in cols), None)
    clu_col = next((cols[k] for k in ['cluster', '군집'] if k in cols), None)

    if lat_col is None or lon_col is None or clu_col is None:
        return None, "quake_points.csv에 위도/경도/cluster 컬럼이 있어야 합니다."

    df = df[[lat_col, lon_col, clu_col]].dropna()
    df.columns = ['lat', 'lon', 'cluster']
    df['cluster'] = df['cluster'].astype(int)

    # 너무 많으면 샘플링 (성능)
    if len(df) > 8000:
        df = df.sample(8000, random_state=42)

    return df, None

quakes, err = load_quakes()

if err:
    st.error(err + " 노트북에서 `df_plot_data.to_csv('quake_points.csv', index=False, encoding='utf-8-sig')` 를 실행해 같은 폴더에 저장해주세요.")
    st.stop()

# 군집 의미 매핑 (노트북: 0=중간, 1=낮음, 2=높음)
CLUSTER_INFO = [
    {"name": "중간 위험",  "tone": "warn",   "color": "#ffd166", "desc": "규모와 영향도가 중간 수준이며 진원이 깊은 지진 패턴입니다."},
    {"name": "낮은 위험",  "tone": "safe",   "color": "#6bcf9f", "desc": "규모와 영향도가 낮은 안전한 지진 패턴입니다."},
    {"name": "높은 위험",  "tone": "danger", "color": "#ff6b6b", "desc": "규모가 크고 진원이 얕은 위험한 지진 패턴입니다."},
]

payload = {
    "points": quakes.values.tolist(),  # [[lat, lon, cluster], ...]
    "clusterInfo": CLUSTER_INFO,
}
payload_json = json.dumps(payload)

html = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet" />
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { box-sizing: border-box; font-family: 'Inter', sans-serif; }
body { margin: 0; padding: 0; background: transparent; color: #fff; }

.workspace {
    display: grid;
    grid-template-columns: 1.7fr 1fr;
    gap: 14px;
    width: 100%;
    height: 620px;
}

.panel {
    position: relative;
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(24px) saturate(140%);
    -webkit-backdrop-filter: blur(24px) saturate(140%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    display: flex;
    flex-direction: column;
}
.panel-label {
    color: rgba(255,255,255,0.5);
    font-size: 0.64rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    padding: 14px 18px 0 18px;
}

/* === MAP === */
#map {
    flex: 1;
    width: 100%;
    border-radius: 0 0 20px 20px;
    position: relative;
}
.maplibregl-canvas { outline: none; }
.maplibregl-ctrl-attrib {
    background: rgba(0,0,0,0.4) !important;
    color: rgba(255,255,255,0.5) !important;
    font-size: 9px !important;
}
.maplibregl-ctrl-attrib a { color: rgba(255,255,255,0.7) !important; }

/* User marker pulse */
.user-marker {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #fff;
    box-shadow: 0 0 0 4px rgba(255,255,255,0.25), 0 0 22px rgba(255,255,255,0.7);
    position: relative;
    cursor: pointer;
}
.user-marker::after {
    content: '';
    position: absolute;
    inset: -6px;
    border-radius: 50%;
    border: 2px solid rgba(255,255,255,0.6);
    animation: pulse 1.6s ease-out infinite;
}
@keyframes pulse {
    0% { transform: scale(0.8); opacity: 0.9; }
    100% { transform: scale(2.4); opacity: 0; }
}

/* === SIDE PANEL === */
.side-content {
    padding: 18px 20px 20px 20px;
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
}
.side-section { margin-top: 14px; }
.section-title {
    color: rgba(255,255,255,0.5);
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 12px;
}

.coord-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}
.coord-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 10px 14px;
}
.coord-box .l {
    color: rgba(255,255,255,0.4);
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 600;
}
.coord-box .v {
    color: #fff;
    font-size: 1.1rem;
    font-weight: 700;
    margin-top: 4px;
    letter-spacing: -0.02em;
}

.hint {
    color: rgba(255,255,255,0.38);
    font-size: 0.74rem;
    margin: 4px 0 12px 0;
    line-height: 1.5;
}

/* Result */
.result-block {
    margin-top: 18px;
    padding: 18px 16px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    text-align: center;
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.result-block.has-result { border-color: rgba(255,255,255,0.12); }
.r-label {
    color: rgba(255,255,255,0.42);
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.2em;
}
.r-cluster {
    color: rgba(255,255,255,0.4);
    font-size: 0.72rem;
    margin: 14px 0 6px 0;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-weight: 700;
}
.r-name {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1.05;
    margin: 0;
    color: rgba(255,255,255,0.5);
    transition: color 0.4s ease;
}
.r-name.safe { color: #6bcf9f; }
.r-name.warn { color: #ffd166; }
.r-name.danger { color: #ff6b6b; }
.r-desc {
    color: rgba(255,255,255,0.55);
    font-size: 0.78rem;
    line-height: 1.55;
    margin: 12px auto 0 auto;
    max-width: 280px;
}
.r-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    padding-top: 14px;
    margin-top: 14px;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.r-stat .v {
    color: #fff;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.r-stat .l {
    color: rgba(255,255,255,0.35);
    font-size: 0.55rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 2px;
    font-weight: 600;
}
.r-stat .dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 4px;
    vertical-align: middle;
    transform: translateY(-1px);
}
.r-stat.s .dot { background: #6bcf9f; box-shadow: 0 0 8px rgba(107,207,159,0.7); }
.r-stat.w .dot { background: #ffd166; box-shadow: 0 0 8px rgba(255,209,102,0.7); }
.r-stat.d .dot { background: #ff6b6b; box-shadow: 0 0 8px rgba(255,107,107,0.7); }

/* Legend on map */
.legend {
    position: absolute;
    bottom: 14px;
    left: 14px;
    background: rgba(8,9,12,0.78);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 10px 14px;
    z-index: 1;
    pointer-events: none;
}
.legend-title {
    color: rgba(255,255,255,0.5);
    font-size: 0.55rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-weight: 700;
    margin-bottom: 6px;
}
.legend-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.8);
    margin-top: 4px;
}
.legend-row .swatch {
    width: 9px; height: 9px;
    border-radius: 50%;
}
</style>
</head>
<body>

<div class="workspace">
    <div class="panel">
        <div class="panel-label">Global Seismic Map · Click anywhere</div>
        <div id="map"></div>
        <div class="legend">
            <div class="legend-title">Cluster Legend</div>
            <div class="legend-row"><span class="swatch" style="background:#6bcf9f;box-shadow:0 0 8px #6bcf9f;"></span> 낮은 위험 (Cluster 1)</div>
            <div class="legend-row"><span class="swatch" style="background:#ffd166;box-shadow:0 0 8px #ffd166;"></span> 중간 위험 (Cluster 0)</div>
            <div class="legend-row"><span class="swatch" style="background:#ff6b6b;box-shadow:0 0 8px #ff6b6b;"></span> 높은 위험 (Cluster 2)</div>
        </div>
    </div>

    <div class="panel">
        <div class="side-content">
            <div class="panel-label" style="padding:0;">Location Input</div>
            <div class="hint">지도를 클릭하거나 위경도를 직접 입력하세요.</div>

            <div class="coord-grid">
                <div class="coord-box">
                    <div class="l">Latitude</div>
                    <div class="v" id="d-lat">—</div>
                </div>
                <div class="coord-box">
                    <div class="l">Longitude</div>
                    <div class="v" id="d-lon">—</div>
                </div>
            </div>

            <div class="result-block" id="result-block">
                <div class="r-label">Seismic Risk Assessment</div>
                <div class="r-cluster" id="r-cluster">Cluster —</div>
                <h2 class="r-name" id="r-name">대기 중</h2>
                <p class="r-desc" id="r-desc">지도에서 지점을 선택하면 반경 ±5° 안의 지진 데이터를 분석해 위험도를 예측합니다.</p>
                <div class="r-stats" id="r-stats" style="display:none;">
                    <div class="r-stat s"><div class="v"><span class="dot"></span><span id="c-safe">0</span></div><div class="l">낮음</div></div>
                    <div class="r-stat w"><div class="v"><span class="dot"></span><span id="c-warn">0</span></div><div class="l">중간</div></div>
                    <div class="r-stat d"><div class="v"><span class="dot"></span><span id="c-danger">0</span></div><div class="l">높음</div></div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
var DATA = __PAYLOAD__;
var POINTS = DATA.points; // [lat, lon, cluster]
var CLUSTER_INFO = DATA.clusterInfo;

// 군집 색 매핑 (노트북: 0=중간, 1=낮음, 2=높음)
var CLUSTER_COLOR = ['#ffd166', '#6bcf9f', '#ff6b6b'];

// MapLibre 초기화 - CartoDB Dark Matter (라벨 적은 다크 베이스맵)
var map = new maplibregl.Map({
    container: 'map',
    style: {
        version: 8,
        sources: {
            'carto-dark': {
                type: 'raster',
                tiles: [
                    'https://a.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png',
                    'https://b.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png',
                    'https://c.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png',
                    'https://d.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png'
                ],
                tileSize: 256,
                attribution: '© OpenStreetMap contributors © CARTO'
            }
        },
        layers: [{
            id: 'carto-dark-layer',
            type: 'raster',
            source: 'carto-dark'
        }]
    },
    center: [20, 15],
    zoom: 1.4,
    minZoom: 1,
    maxZoom: 8,
    attributionControl: true
});

map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');

// 지진 데이터를 GeoJSON으로 변환
var features = POINTS.map(function(p) {
    return {
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [p[1], p[0]] }, // [lon, lat]
        properties: { cluster: p[2] }
    };
});

map.on('load', function() {
    map.addSource('quakes', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: features }
    });

    // 글로우 후광 레이어 (블러된 큰 원)
    map.addLayer({
        id: 'quakes-glow',
        type: 'circle',
        source: 'quakes',
        paint: {
            'circle-radius': [
                'interpolate', ['linear'], ['zoom'],
                1, 4, 4, 10, 7, 22
            ],
            'circle-color': [
                'match', ['get', 'cluster'],
                0, '#ffd166',
                1, '#6bcf9f',
                2, '#ff6b6b',
                '#888'
            ],
            'circle-opacity': 0.18,
            'circle-blur': 1.2
        }
    });

    // 메인 도트 레이어
    map.addLayer({
        id: 'quakes-core',
        type: 'circle',
        source: 'quakes',
        paint: {
            'circle-radius': [
                'interpolate', ['linear'], ['zoom'],
                1, 1.5, 4, 3, 7, 5
            ],
            'circle-color': [
                'match', ['get', 'cluster'],
                0, '#ffd166',
                1, '#6bcf9f',
                2, '#ff6b6b',
                '#aaa'
            ],
            'circle-opacity': 0.95,
            'circle-stroke-width': 0.5,
            'circle-stroke-color': 'rgba(255,255,255,0.5)'
        }
    });
});

// 사용자 마커
var userMarker = null;

function predictRisk(lat, lon) {
    // 반경 ±5° 안의 군집 카운트
    var counts = [0, 0, 0];
    for (var i = 0; i < POINTS.length; i++) {
        var plat = POINTS[i][0], plon = POINTS[i][1], pc = POINTS[i][2];
        if (Math.abs(plat - lat) <= 5 && Math.abs(plon - lon) <= 5) {
            counts[pc]++;
        }
    }
    var total = counts[0] + counts[1] + counts[2];
    if (total === 0) return { cluster: -1, counts: counts, total: 0 };

    var bestIdx = 0, bestCnt = counts[0];
    for (var k = 1; k < 3; k++) {
        if (counts[k] > bestCnt) { bestCnt = counts[k]; bestIdx = k; }
    }
    return { cluster: bestIdx, counts: counts, total: total };
}

function updatePanel(lat, lon) {
    document.getElementById('d-lat').textContent = lat.toFixed(2) + '°';
    document.getElementById('d-lon').textContent = lon.toFixed(2) + '°';

    var res = predictRisk(lat, lon);
    var nameEl = document.getElementById('r-name');
    var clusterEl = document.getElementById('r-cluster');
    var descEl = document.getElementById('r-desc');
    var statsEl = document.getElementById('r-stats');
    var block = document.getElementById('result-block');

    if (res.total === 0) {
        nameEl.textContent = '데이터 없음';
        nameEl.className = 'r-name';
        clusterEl.textContent = '반경 ±5° 안 지진 0건';
        descEl.textContent = '주변에 분석할 지진 데이터가 없습니다. 다른 지점을 선택해보세요.';
        statsEl.style.display = 'none';
        return;
    }

    var info = CLUSTER_INFO[res.cluster];
    nameEl.textContent = info.name;
    nameEl.className = 'r-name ' + info.tone;
    clusterEl.textContent = 'Cluster · ' + res.cluster + ' · 주변 ' + res.total + '건';
    descEl.textContent = info.desc;

    statsEl.style.display = 'grid';
    document.getElementById('c-safe').textContent = res.counts[1];   // 낮음 = cluster 1
    document.getElementById('c-warn').textContent = res.counts[0];   // 중간 = cluster 0
    document.getElementById('c-danger').textContent = res.counts[2]; // 높음 = cluster 2
    block.classList.add('has-result');
}

function placeUserMarker(lat, lon) {
    if (userMarker) userMarker.remove();
    var el = document.createElement('div');
    el.className = 'user-marker';
    userMarker = new maplibregl.Marker({ element: el })
        .setLngLat([lon, lat])
        .addTo(map);
    updatePanel(lat, lon);
}

map.on('click', function(e) {
    placeUserMarker(e.lngLat.lat, e.lngLat.lng);
});

// 초기 데모: 일본 근처
map.on('load', function() {
    setTimeout(function() { placeUserMarker(35.6, 139.7); }, 400);
});
</script>

</body>
</html>
"""

html = html.replace("__PAYLOAD__", payload_json)
components.html(html, height=640, scrolling=False)