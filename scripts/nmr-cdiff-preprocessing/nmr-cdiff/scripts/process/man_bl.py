from operator import sub

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
import scipy

def get_aspect(ax):
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio

class Baseline:

    def __init__(self, spec_ppm, spec_signal, nl, order, nr=0.15):
        self.figure, self.ax = plt.subplots()
        self.figure.subplots_adjust(bottom=0.2)
        self.spec = self.ax.plot(spec_ppm, spec_signal)[0]
        self.baseline = self.ax.plot(spec_ppm, np.zeros_like(spec_signal))[0]
        self.order = order
        self.nodes = {}
        self.node_radius = nr
        self.node_vscale = 1./get_aspect(self.ax)
        self.selected = None
        for node in nl:
            self._add_node(node)
        
        # Initialize interactive buttons
        axadd = self.figure.add_axes([0.1, 0.05, 0.395, 0.075])
        self.badd = mpl.widgets.Button(axadd, "Add node")
        self.badd.on_clicked(self.add_click)
        axrem = self.figure.add_axes([0.505, 0.05, 0.395, 0.075])
        self.brem = mpl.widgets.Button(axrem, "Remove node")
        self.brem.on_clicked(self.remove_click)

    def _add_node(self, shift):
        """Add node to baseline curve given chemical shift."""
        if len(self.nodes) == 0:
            nid = 0
        else:
            nid = max(self.nodes) + 1
        pos = self.calculate_xy(shift)
        node_patch = mpl.patches.Ellipse(
            pos, 
            width=2*self.node_radius,
            height=2*self.node_radius*self.node_vscale,
            facecolor='b',
            zorder=4,
        )
        self.ax.add_artist(node_patch)
        node = BaselineNode(nid, node_patch, self)
        node.connect()
        self.nodes[nid] = node
        self.update_baseline()
        node.select()

    def calculate_xy(self, shift):
        """Calculate (x, y) coordinates for node given chemical shift.
        
        xx will be nearest ppm shift value in the dataset
        yy will be the average signal of all points within window of the center
        """
        spec_ppm = self.spec.get_xdata()
        i = np.abs(spec_ppm - shift).argmin()
        xx = spec_ppm[i]
        ymask = (spec_ppm < (shift + self.node_radius)) \
                & (spec_ppm > (shift - self.node_radius))
        yy = self.spec.get_ydata()[ymask].mean()
        return xx, yy

    def add_click(self, event):
        self.add_node()

    def remove_click(self, event):
        self.remove_node()

    def add_node(self):
        """Add node to baseline curve after selected node, or in center of spectrum."""
        sorted_ids = [e[0] for e in sorted(self.nodes.items(), key=lambda x: x[1].x())]
        if (self.selected is not None
                and self.selected.id != sorted_ids[-1]):
            p = self.selected.id
            q = sorted_ids[sorted_ids.index(p) + 1]
            newshift = (self.nodes[p].x() + self.nodes[q].x())/2
        else:
            xdata = self.spec.get_xdata()
            newshift = xdata[len(xdata) // 2]
        self._add_node(newshift)

    def remove_node(self):
        if self.selected is not None:
            self.selected.remove()

    def get_baseline(self):
        return self.baseline.get_ydata()

    def update_baseline(self):
        """Re-fit and render new baseline."""
        if len(self.nodes) > self.order + 1:
            # fit_poly = Polynomial.fit(
            #     *zip(*[n.node.get_center() for n in self.nodes.values()]), 
            #     deg=self.order
            # )
            # self.baseline.set_ydata(fit_poly(self.spec.get_xdata()))
            fit_spline = scipy.interpolate.UnivariateSpline(
                *zip(*[n.node.get_center() for n in self.nodes.values()]),
                k=5,
            )
            self.baseline.set_ydata(fit_spline(self.spec.get_xdata()))
            self.figure.canvas.draw()

class BaselineNode:
    lock = None

    def __init__(self, id, patch, parent):
        self.id = id
        self.node = patch
        self.parent = parent
        self.press = None
        self.background = None

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.parent.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.parent.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.parent.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def select(self):
        if self.parent.selected is not None:
            self.parent.selected.deselect()
        self.parent.selected = self
        self.node.set(edgecolor='r', lw=0.5)
        self.parent.figure.canvas.draw()

    def deselect(self):
        if self.parent.selected is self:
            self.parent.selected = None
            self.node.set(edgecolor='k', lw=0.1)
            self.parent.figure.canvas.draw()

    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        if (event.inaxes != self.node.axes
                or BaselineNode.lock is not None):
            return
        contains, attrd = self.node.contains(event)
        if not contains:
            self.deselect()
            return
        # print('event contains', self.node.get_center())
        self.press = self.x(), event.xdata
        BaselineNode.lock = self
        self.select()

        # draw everything but the selected nodeangle and store the pixel buffer
        canvas = self.parent.figure.canvas
        axes = self.node.axes
        self.node.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.node.axes.bbox)

        # now redraw just the nodeangle
        axes.draw_artist(self.node)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        """Move the nodeangle if the mouse is over us."""
        if (event.inaxes != self.node.axes
                or BaselineNode.lock is not self):
            return
        x0, xpress = self.press
        dx = event.xdata - xpress
        pos = self.parent.calculate_xy(x0 + dx)
        self.node.set_center(pos)

        canvas = self.parent.figure.canvas
        axes = self.parent.ax
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current nodeangle
        axes.draw_artist(self.node)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        """Clear button press information."""
        if BaselineNode.lock is not self:
            return

        self.press = None
        BaselineNode.lock = None

        # turn off the node animation property and reset the background
        self.node.set_animated(False)
        self.background = None

        # redraw the full figure
        self.parent.update_baseline()
        self.parent.figure.canvas.draw()

    def disconnect(self):
        """Disconnect all callbacks."""
        self.parent.figure.canvas.mpl_disconnect(self.cidpress)
        self.parent.figure.canvas.mpl_disconnect(self.cidrelease)
        self.parent.figure.canvas.mpl_disconnect(self.cidmotion)

    def remove(self):
        """Delete node."""
        self.deselect()
        self.disconnect()
        self.node.set_visible(False)
        self.parent.nodes.pop(self.id)
        self.parent.update_baseline()
        self.parent = None

    def x(self):
        return self.node.get_center()[0]