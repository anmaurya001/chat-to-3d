"""Status panel component for system monitoring."""

import gradio as gr


def create_status_panel():
    """Create the status monitoring panel."""
    
    with gr.Column(elem_classes=["status-panel"]) as status_section:
        # Panel header with close button
        with gr.Row(elem_classes=["panel-header"]):
            gr.Markdown("### Operations Status")
            close_panel_btn = gr.Button("✕", elem_classes=["close-panel-btn"], size="sm")
        
        # Status indicators
        llm_status = gr.HTML("""
            <div class="status-item">
                <span class="status-indicator online"></span>
                <span>LLM Agent: Online</span>
            </div>
        """)
        
        generation_status = gr.HTML("""
            <div class="status-item">
                <span class="status-indicator idle"></span>
                <span>3D Generation: Idle</span>
            </div>
        """)
        
        # Refresh button
        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
        
        # Status log
        status_log = gr.Textbox(
            label="Status Log",
            lines=8,
            max_lines=20,
            interactive=False,
            value="System initialized...\n"
        )
    
    return {
        "section": status_section,
        "close_btn": close_panel_btn,
        "llm_status": llm_status,
        "generation_status": generation_status,
        "refresh_btn": refresh_btn,
        "log": status_log
    } 