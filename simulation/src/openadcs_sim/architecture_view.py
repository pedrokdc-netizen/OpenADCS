import math
import tkinter as tk
from tkinter import ttk

from .architecture import CONNECTIONS, DIAGRAM_SIZE, MODULE_BY_ID, MODULES, ModuleStatus


PALETTE = {
    ModuleStatus.IMPLEMENTED: ("#dbeafe", "#1d4ed8"),
    ModuleStatus.PARTIAL: ("#fef3c7", "#92400e"),
    ModuleStatus.PLACEHOLDER: ("#f1f5f9", "#475569"),
}
SHORT_TITLES = {
    "orbit": "Orbit", "magnetic": "Magnetics", "sun": "Sun",
    "conditions": "Env inputs", "dynamics": "Dynamics", "spacecraft": "S/C config",
    "sunsensor": "Sun cam x6", "gyro": "Gyro", "horizon": "Horizon", "gps": "GPS / GNSS", "mag": "TAM",
    "adcs": "ADCS MCU", "obc": "OBC MCU", "actuators": "Actuators",
}
SHORT_VARIABLES = {
    "sun_sunsensor": "s_ITRS", "sunsensor_adcs": "fused s_body",
    "orbit_gps": "r, v, UTC",
    "dynamics_gyro": "omega_B",
    "orbit_horizon": "r_ITRS", "dynamics_horizon": "R_BG",
    "conditions_gps": "SV / RF / DOP", "conditions_sensors": "sunlit / albedo",
    "conditions_gyro": "T", "magnetic_reference": "B_ref",
    "sun_reference": "s_ref", "gyro_adcs": "omega_B",
    "horizon_adcs": "nadir_body", "gps_adcs": "ECEF PVT",
    "adcs_obc": "health / estimate", "obc_adcs": "mode / target / clock",
    "adcs_actuators": "torque_cmd", "actuators_dynamics": "torque_body",
    "orbit_magnetic": "r, UTC", "orbit_sun": "r, UTC",
    "dynamics_sunsensor": "R_BG",
    "spacecraft_dynamics": "m, I_B",
    "magnetic_mag": "B_ref -> B_body", "dynamics_mag": "R_BG", "mag_adcs": "B_meas",
}


def nearest_on_route(points, location):
    candidates = []
    for first, second in zip(points, points[1:]):
        x = min(max(location[0], min(first[0], second[0])), max(first[0], second[0]))
        y = min(max(location[1], min(first[1], second[1])), max(first[1], second[1]))
        candidates.append((x, y))
    return min(candidates, key=lambda point: math.dist(point, location))


class ArchitecturePanel(ttk.Frame):
    """Read-only interface map; never invokes the represented modules."""

    def __init__(self, parent):
        super().__init__(parent, padding=8)
        self.scale = 1.0
        self.fit_mode = True
        self.selected_module: str | None = None
        self.selected_connection: str | None = None
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X)
        for text, command in (
            ("Fit diagram", self.fit), ("100%", lambda: self.set_zoom(1.0)),
            ("Zoom -", lambda: self.set_zoom(self.scale / 1.25)),
            ("Zoom +", lambda: self.set_zoom(self.scale * 1.25)),
        ):
            ttk.Button(toolbar, text=text, command=command).pack(side=tk.LEFT, padx=(0, 4))
        self.zoom_text = tk.StringVar(value="100%")
        ttk.Label(toolbar, textvariable=self.zoom_text, width=6).pack(side=tk.LEFT)
        self.module_choice = ttk.Combobox(
            toolbar, state="readonly", width=26,
            values=[module.title for module in MODULES],
        )
        self.module_choice.pack(side=tk.RIGHT)
        self.module_choice.bind("<<ComboboxSelected>>", self._choose_module)
        ttk.Label(
            self, text="Static interface map, NOT live execution. Click a block/link for contracts; zoom for full labels.",
            foreground="#475569", wraplength=650,
        ).pack(anchor=tk.W, pady=(6, 2))
        ttk.Label(
            self, text="Blue: implemented  |  Amber: partial  |  Gray: PLACEHOLDER  |  Solid: available API/offline path  |  Dashed: planned",
            foreground="#334155", wraplength=650,
        ).pack(anchor=tk.W, pady=(0, 6))
        drawing = ttk.Frame(self)
        drawing.pack(fill=tk.BOTH, expand=True)
        drawing.rowconfigure(0, weight=1)
        drawing.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(drawing, background="#ffffff", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        horizontal = ttk.Scrollbar(drawing, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vertical = ttk.Scrollbar(drawing, orient=tk.VERTICAL, command=self.canvas.yview)
        horizontal.grid(row=1, column=0, sticky="ew")
        vertical.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.canvas.bind("<Configure>", self._resize)
        self.canvas.bind("<ButtonPress-2>", lambda event: self.canvas.scan_mark(event.x, event.y))
        self.canvas.bind("<B2-Motion>", lambda event: self.canvas.scan_dragto(event.x, event.y, gain=1))
        details_frame = ttk.Frame(self)
        details_frame.pack(fill=tk.X, pady=(6, 0))
        self.details = tk.Text(details_frame, height=7, wrap=tk.WORD, font=("Consolas", 10), relief=tk.FLAT)
        details_scroll = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.details.yview)
        details_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.details.configure(yscrollcommand=details_scroll.set)
        self.details.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._details(
            "Available before running a simulation. This view documents existing and proposed interfaces only.\n"
            "PARTIAL means existing processing/ingress is not a complete digital model or MCU emulator.\n"
            "PLACEHOLDER modules raise NotImplementedError; no sensor truth, torque or attitude is synthesized.\n"
            "Fit shows compact variable labels. Use 100% / Zoom + for full variables and units.\n"
            "Use scrollbars or middle-button drag to pan; selecting a module at 100% centers it.",
        )
        self.draw()

    def _details(self, text):
        self.details.configure(state=tk.NORMAL)
        self.details.delete("1.0", tk.END)
        self.details.insert("1.0", text)
        self.details.configure(state=tk.DISABLED)

    def _resize(self, event):
        if self.fit_mode and event.width > 100 and event.height > 100:
            self.fit(event.width, event.height)

    def fit(self, width=None, height=None):
        width = self.canvas.winfo_width() if width is None else width
        height = self.canvas.winfo_height() if height is None else height
        if width <= 100 or height <= 100:
            return
        self.fit_mode = True
        self.scale = max(0.25, min((width - 12) / DIAGRAM_SIZE[0], (height - 12) / DIAGRAM_SIZE[1], 1.0))
        self.draw()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def set_zoom(self, scale):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Diagram zoom must be finite and positive")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        center = (
            self.canvas.canvasx(width / 2) / self.scale,
            self.canvas.canvasy(height / 2) / self.scale,
        )
        self.fit_mode = False
        self.scale = min(max(scale, 0.25), 2.0)
        self.draw()
        self._center(*center)

    def _center(self, x, y):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.xview_moveto(max(0, (x * self.scale - width / 2) / (DIAGRAM_SIZE[0] * self.scale)))
        self.canvas.yview_moveto(max(0, (y * self.scale - height / 2) / (DIAGRAM_SIZE[1] * self.scale)))

    def _choose_module(self, _event):
        index = self.module_choice.current()
        if index >= 0:
            self.select_module(MODULES[index].id)
            self.set_zoom(1.0)
            bounds = MODULES[index].bounds
            self._center((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)

    def select_module(self, module_id):
        module = MODULE_BY_ID[module_id]
        self.selected_module, self.selected_connection = module_id, None
        self.module_choice.set(module.title)
        self._details(
            f"{module.title} | {module.status.value}\n{module.summary}\n"
            f"Inputs: {'; '.join(module.inputs)}\nOutputs: {'; '.join(module.outputs)}\n"
            f"Source / boilerplate: {'; '.join(module.source_paths)}\n"
            "This selection does not enable or execute a module.",
        )
        self.draw()

    def select_connection(self, connection_id):
        connection = next(item for item in CONNECTIONS if item.id == connection_id)
        self.selected_connection, self.selected_module = connection_id, None
        self._details(
            f"{MODULE_BY_ID[connection.source].title} -> {MODULE_BY_ID[connection.target].title}\n"
            f"Variables: {connection.variables}\n"
            f"Interface: {'PLANNED / not connected' if connection.planned else 'Available API or offline path'}\n"
            "Arrows describe contracts, not active bus traffic. MCU timing and hardware transport are not simulated.",
        )
        self.draw()

    def draw(self):
        canvas, scale = self.canvas, self.scale
        canvas.delete("all")
        self.zoom_text.set(f"{scale * 100:.0f}%")
        canvas.configure(scrollregion=(0, 0, DIAGRAM_SIZE[0] * scale, DIAGRAM_SIZE[1] * scale))
        compact = scale < 0.75
        for connection in CONNECTIONS:
            selected = connection.id == self.selected_connection or (
                self.selected_module in (connection.source, connection.target)
            )
            color = "#c2410c" if selected else "#94a3b8" if connection.planned else "#2563eb"
            tag = f"connection:{connection.id}"
            coordinates = [value * scale for point in connection.points for value in point]
            canvas.create_line(
                *coordinates, fill=color, width=3 if selected else 1.5,
                dash=(5, 4) if connection.planned else (),
                arrow=tk.LAST, arrowshape=(8, 10, 4), tags=(tag, "connection"),
            )
            nearest = nearest_on_route(connection.points, connection.label_position)
            if math.dist(nearest, connection.label_position) > 32:
                canvas.create_line(
                    *(value * scale for point in (nearest, connection.label_position) for value in point),
                    fill=color, dash=(2, 3), tags=(tag, "leader"),
                )
            label = canvas.create_text(
                *(value * scale for value in connection.label_position),
                text=SHORT_VARIABLES[connection.id] if compact else connection.variables,
                width=340 * scale, font=("Segoe UI", max(6, round(10 * scale))),
                fill="#9a3412" if selected else "#334155", tags=(tag, "connection_label"),
            )
            bounds = canvas.bbox(label)
            if bounds is not None:
                background = canvas.create_rectangle(
                    bounds[0] - 2, bounds[1] - 1, bounds[2] + 2, bounds[3] + 1,
                    fill="white", outline="", tags=(tag, "label_background"),
                )
                canvas.tag_lower(background, label)
            canvas.tag_bind(tag, "<Button-1>", lambda _event, key=connection.id: self.select_connection(key))
        for module in MODULES:
            x1, y1, x2, y2 = (value * scale for value in module.bounds)
            fill, outline = PALETTE[module.status]
            tag = f"module:{module.id}"
            canvas.create_rectangle(
                x1, y1, x2, y2, fill=fill,
                outline="#c2410c" if module.id == self.selected_module else outline,
                width=3 if module.id == self.selected_module else 1.5,
                dash=(5, 3) if module.status is ModuleStatus.PLACEHOLDER else (),
                tags=(tag, "module_box"),
            )
            caption = SHORT_TITLES[module.id] if compact else module.title
            font_size = max(7, round(12 * scale))
            title_y = y1 + (y2 - y1) * 0.27 if compact else y1 + 20 * scale
            status_y = y1 + (y2 - y1) * 0.77 if compact else y1 + 42 * scale
            canvas.create_text(
                (x1 + x2) / 2, title_y, text=caption, width=(x2 - x1) - 8,
                font=("Segoe UI", font_size, "bold"), fill="#0f172a", tags=(tag, "module_title"),
            )
            canvas.create_text(
                (x1 + x2) / 2, status_y, text=module.status.value,
                font=("Segoe UI", max(6, round(9 * scale)), "bold"),
                fill=outline, tags=(tag, "module_status"),
            )
            if scale >= 1.0:
                canvas.create_text(
                    (x1 + x2) / 2, y1 + 59 * scale, anchor=tk.N,
                    text=module.summary, width=(x2 - x1) - 16 * scale,
                    font=("Segoe UI", max(7, round(10 * scale))), fill="#334155",
                    tags=(tag, "module_summary"),
                )
            canvas.tag_bind(tag, "<Button-1>", lambda _event, key=module.id: self.select_module(key))
