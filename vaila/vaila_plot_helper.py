import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import json
import base64
from io import BytesIO
from matplotlib.lines import Line2D
import numpy as np

def generate_visual_report(analyzer, out_dir, name: str, fps: float, report_data: dict):
    plt.switch_backend('Agg')
    com = analyzer.df[['CoM_x', 'CoM_y', 'CoM_z']].to_numpy()
    t = np.arange(len(com)) / fps
    
    def get_base64_image(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
        
    images_b64 = []
    
    # --- Plot 1: Mixed XYZ + Events ---
    fig1 = plt.figure(figsize=(16, 6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(t, com[:, 0], label='X (Medio-Lateral)', color='blue', linewidth=2)
    ax1.plot(t, com[:, 1], label='Y (Anteroposterior)', color='orange', linewidth=2)
    ax1.plot(t, com[:, 2], label='Z (Vertical)', color='purple', linewidth=2)
    
    for leg, color in [('Right', 'red'), ('Left', 'green')]:
        for ev, ls in [('HS', '--'), ('TO', ':')]:
            frames = analyzer.gait_events.get(leg, {}).get(ev, [])
            for i, f in enumerate(frames):
                label = f"{leg} {ev}" if i == 0 else ""
                ax1.axvline(x=f/fps, color=color, linestyle=ls, alpha=0.6, label=label)
    ax1.set_title('CoM Trajectory (X, Y, Z) vs Time + Gait Events')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True)
    images_b64.append(get_base64_image(fig1))
    
    # --- Plot 2: Split XYZ ---
    fig2 = plt.figure(figsize=(16, 4))
    gs2 = gridspec.GridSpec(1, 3, figure=fig2)
    ax_x = fig2.add_subplot(gs2[0, 0])
    ax_x.plot(t, com[:, 0], color='blue')
    ax_x.set_title('CoM X (Medio-Lateral)')
    ax_x.grid(True)
    ax_y = fig2.add_subplot(gs2[0, 1])
    ax_y.plot(t, com[:, 1], color='orange')
    ax_y.set_title('CoM Y (Anteroposterior)')
    ax_y.grid(True)
    ax_z = fig2.add_subplot(gs2[0, 2])
    ax_z.plot(t, com[:, 2], color='purple')
    ax_z.set_title('CoM Z (Vertical)')
    ax_z.grid(True)
    plt.tight_layout()
    images_b64.append(get_base64_image(fig2))
    
    # --- Plot 3: Stick figures Sagittal ---
    def draw_stick_figures(ax, frames, title):
        for f in frames:
            pts = {}
            for p_idx in range(11, 33):
                try:
                    pts[p_idx] = analyzer._get_point_3d(p_idx)[f]
                except ValueError:
                    continue
            
            lines = [
                ([12, 14], 'red'), ([14, 16], 'red'), ([12, 24], 'red'), ([24, 26], 'red'), ([26, 28], 'red'), ([28, 30], 'red'), ([30, 32], 'red'), ([28, 32], 'red'),
                ([11, 13], 'blue'), ([13, 15], 'blue'), ([11, 23], 'blue'), ([23, 25], 'blue'), ([25, 27], 'blue'), ([27, 29], 'blue'), ([29, 31], 'blue'), ([27, 31], 'blue'),
                ([11, 12], 'black'), ([23, 24], 'black')
            ]
            for pair, color in lines:
                if pair[0] in pts and pair[1] in pts:
                    ax.plot([pts[pair[0]][1], pts[pair[1]][1]], [pts[pair[0]][2], pts[pair[1]][2]], color=color, linewidth=2, alpha=0.7)
                    ax.scatter([pts[pair[0]][1], pts[pair[1]][1]], [pts[pair[0]][2], pts[pair[1]][2]], color=color, s=15, zorder=3)
        ax.set_title(title)
        ax.set_xlabel('Y (Anteroposterior) [m]')
        ax.set_ylabel('Z (Vertical) [m]')
        ax.axis('equal')
        ax.grid(True)
        ax.legend([Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='blue', lw=2)], ['Right Side', 'Left Side'], loc='upper right')

    events = []
    for leg in ['Right', 'Left']:
        for ev in ['HS', 'TO']:
            for f in analyzer.gait_events.get(leg, {}).get(ev, []):
                events.append(int(round(f)))
    events.sort()
    
    phases = report_data.get('Phases_Seconds', {})
    wf_s, wf_e = phases.get('Walk_Forward', (0,0))
    wb_s, wb_e = phases.get('Walk_Back', (0,0))
    
    fig3 = plt.figure(figsize=(16, 5))
    draw_stick_figures(fig3.add_subplot(1, 2, 1), [f for f in events if wf_s*fps <= f <= wf_e*fps], "Walk Forward (Ida) Sagittal Events (Y-Z)")
    draw_stick_figures(fig3.add_subplot(1, 2, 2), [f for f in events if wb_s*fps <= f <= wb_e*fps], "Walk Back (Volta) Sagittal Events (Y-Z)")
    plt.tight_layout()
    images_b64.append(get_base64_image(fig3))
    
    # --- Plot 4: 3D Equal Aspect Plot ---
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111, projection='3d')
    ax4.plot(com[:, 0], com[:, 1], com[:, 2], color='darkred', alpha=0.8, linewidth=2, label='CoM Trajectory')
    ax4.scatter(com[0, 0], com[0, 1], com[0, 2], color='green', s=100, label='Start')
    ax4.scatter(com[-1, 0], com[-1, 1], com[-1, 2], color='red', s=100, label='End')
    max_range = np.array([com[:,0].max()-com[:,0].min(), com[:,1].max()-com[:,1].min(), com[:,2].max()-com[:,2].min()]).max() / 2.0
    mid_x, mid_y, mid_z = (com[:,0].max()+com[:,0].min())*0.5, (com[:,1].max()+com[:,1].min())*0.5, (com[:,2].max()+com[:,2].min())*0.5
    ax4.set_xlim(mid_x - max_range, mid_x + max_range)
    ax4.set_ylim(mid_y - max_range, mid_y + max_range)
    ax4.set_zlim(mid_z - max_range, mid_z + max_range)
    ax4.set_box_aspect([1, 1, 1])
    ax4.set_title('3D CoM Trajectory (Equal Aspect)')
    ax4.set_xlabel('X (ML)')
    ax4.set_ylabel('Y (AP)')
    ax4.set_zlabel('Z (Vert)')
    ax4.legend()
    images_b64.append(get_base64_image(fig4))

    images_html = '\n'.join([f'<img src="data:image/png;base64,{b64}" alt="TUG Chart" style="max-width: 100%; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;"/>' for b64 in images_b64])
    
    meta_html = ''.join([f'<tr><th>{k}</th><td>{v}</td></tr>' for k,v in report_data.get('Metadata', {}).items()])
    spat_global = ''.join([f'<tr><th>{k.replace("_", " ")}</th><td>{v:.3f}</td></tr>' if isinstance(v, (int, float)) else f'<tr><th>{k.replace("_", " ")}</th><td>{v}</td></tr>' for k,v in report_data.get('Spatiotemporal', {}).get('Global', {}).items()])
    spat_right = ''.join([f'<tr><th>{k.replace("_", " ")}</th><td>{v:.3f}</td></tr>' if isinstance(v, (int, float)) else f'<tr><th>{k.replace("_", " ")}</th><td>{v}</td></tr>' for k,v in report_data.get('Spatiotemporal', {}).get('Right', {}).items()])
    spat_left = ''.join([f'<tr><th>{k.replace("_", " ")}</th><td>{v:.3f}</td></tr>' if isinstance(v, (int, float)) else f'<tr><th>{k.replace("_", " ")}</th><td>{v}</td></tr>' for k,v in report_data.get('Spatiotemporal', {}).get('Left', {}).items()])
    phases_html = ''.join([f'<tr><th>{k.replace("_", " ")}</th><td>{round(v[0],2)}s to {round(v[1],2)}s (Dur: {round(v[1]-v[0],2)}s)</td></tr>' if isinstance(v, (list, tuple)) else f'<tr><th>{k.replace("_", " ")}</th><td>{v}</td></tr>' for k,v in report_data.get('Phases_Seconds', {}).items()])

    html_content = f"""
    <!DOCTYPEhtml>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>TUG Report: {name}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f9f9f9; color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; font-weight: bold; width: 40%; }}
            .section-title {{ border-bottom: 2px solid #3498db; padding-bottom: 5px; color: #2c3e50; margin-top: 40px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>TUG Analysis Report</h1>
            <h2>Subject: {name}</h2>
            
            <h3 class="section-title">Visual Charts</h3>
            {images_html}
            
            <h3 class="section-title">Metadata</h3>
            <table>{meta_html}</table>
            
            <h3 class="section-title">Spatiotemporal Parameters</h3>
            <h4>Global</h4><table>{spat_global}</table>
            <h4>Right Leg</h4><table>{spat_right}</table>
            <h4>Left Leg</h4><table>{spat_left}</table>
            
            <h3 class="section-title">TUG Phases (Seconds)</h3>
            <table>{phases_html}</table>
        </div>
    </body>
    </html>
    """
    report_file = out_dir / f"{name}_tug_report.html"
    report_file.write_text(html_content, encoding='utf-8')
    return report_file
