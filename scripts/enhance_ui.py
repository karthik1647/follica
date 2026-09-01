with open('backend/app/static/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

extra_css = """
        /* Shimmering Skeleton Loading Tiles */
        .skeleton-shimmer {
            background: linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 50%, #f1f5f9 75%);
            background-size: 200% 100%;
            animation: shimmer 1.4s ease-in-out infinite;
            border-radius: var(--radius-md);
        }

        @keyframes shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        .skeleton-gauge-box {
            height: 90px;
            margin-bottom: 20px;
        }

        .skeleton-subscores-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }

        .skeleton-subscore-tile {
            height: 64px;
        }

        .skeleton-card-row {
            height: 48px;
            margin-bottom: 12px;
        }

        /* Visual Shooting Guide Card */
        .shooting-guide-banner {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: var(--radius-lg);
            padding: 16px;
            margin-bottom: 18px;
        }

        .shooting-guide-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }

        .guide-badge {
            font-size: 0.72rem;
            font-weight: 800;
            background: #0284c7;
            color: #ffffff;
            padding: 3px 8px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }

        .guide-subtext {
            font-size: 0.78rem;
            color: var(--text-secondary);
            font-weight: 600;
        }

        .shooting-guide-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 10px;
        }

        .guide-item {
            display: flex;
            align-items: flex-start;
            gap: 8px;
            background: #ffffff;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 10px 8px;
            font-size: 0.76rem;
            line-height: 1.35;
        }

        .guide-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            background: #e0f2fe;
            color: #0369a1;
            font-weight: 800;
            font-size: 0.72rem;
            border-radius: 50%;
            flex-shrink: 0;
        }

        .guide-desc {
            color: var(--text-secondary);
            font-size: 0.72rem;
        }
"""

if 'skeleton-shimmer' not in html:
    html = html.replace('/* Form Controls & Chips */', extra_css + '\n        /* Form Controls & Chips */')

shooting_guide_html = """
                        <!-- Visual Photo Shooting Guide -->
                        <div class="shooting-guide-banner">
                            <div class="shooting-guide-header">
                                <span class="guide-badge">PHOTO CAPTURE RULES</span>
                                <span class="guide-subtext">Follow these 4 steps for optimal accuracy:</span>
                            </div>
                            <div class="shooting-guide-grid">
                                <div class="guide-item">
                                    <span class="guide-num">1</span>
                                    <div><strong>Clean Lens</strong><br><span class="guide-desc">Wipe glass to remove oily smudges.</span></div>
                                </div>
                                <div class="guide-item">
                                    <span class="guide-num">2</span>
                                    <div><strong>Flash ON</strong><br><span class="guide-desc">Illuminates hair roots and parting.</span></div>
                                </div>
                                <div class="guide-item">
                                    <span class="guide-num">3</span>
                                    <div><strong>Expose Parting</strong><br><span class="guide-desc">Part hair to show 1-2 in of scalp.</span></div>
                                </div>
                                <div class="guide-item">
                                    <span class="guide-num">4</span>
                                    <div><strong>10-15 cm</strong><br><span class="guide-desc">Hold steady & tap screen to focus.</span></div>
                                </div>
                            </div>
                        </div>
"""

if 'shooting-guide-banner' not in html:
    html = html.replace('<div id="image-upload-zone"', shooting_guide_html + '\n                        <div id="image-upload-zone"')

skeleton_loading_html = """
                    <!-- Shimmering Skeleton Loading Tiles -->
                    <div id="results-skeleton-state" class="app-card" style="display: none;">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #0284c7; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between;">
                            <span>Analyzing Scalp Features...</span>
                            <span style="font-size: 0.76rem; color: #64748b;">Dual-Pipeline (DL + CV)</span>
                        </div>
                        <div class="skeleton-shimmer skeleton-gauge-box"></div>
                        <div class="skeleton-subscores-grid">
                            <div class="skeleton-shimmer skeleton-subscore-tile"></div>
                            <div class="skeleton-shimmer skeleton-subscore-tile"></div>
                            <div class="skeleton-shimmer skeleton-subscore-tile"></div>
                            <div class="skeleton-shimmer skeleton-subscore-tile"></div>
                        </div>
                        <div class="skeleton-shimmer skeleton-card-row"></div>
                        <div class="skeleton-shimmer skeleton-card-row"></div>
                        <div class="skeleton-shimmer skeleton-card-row"></div>
                    </div>
"""

if 'results-skeleton-state' not in html:
    html = html.replace('<div id="results-placeholder"', skeleton_loading_html + '\n                    <div id="results-placeholder"')

html = html.replace(
    'document.getElementById("loading-spinner").style.display = "block";',
    'document.getElementById("loading-spinner").style.display = "block";\n            document.getElementById("results-placeholder").style.display = "none";\n            document.getElementById("results-skeleton-state").style.display = "block";\n            document.getElementById("results-panel-card").style.display = "none";'
)

html = html.replace(
    'document.getElementById("loading-spinner").style.display = "none";',
    'document.getElementById("loading-spinner").style.display = "none";\n            document.getElementById("results-skeleton-state").style.display = "none";'
)

with open('backend/app/static/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Updated index.html with skeleton tiles and shooting guide successfully')