/*
 * ==============================================================================
 * Script: freekiki_report_app.js - interactive anchor editor of the FreeKiki press report
 * ==============================================================================
 *
 * vailá - Multimodal Toolbox
 * © Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
 * https://github.com/vaila-multimodaltoolbox/vaila
 * Please see AUTHORS for contributors.
 *
 * Author: Paulo Roberto Pereira Santiago
 * Email: paulosantiago@usp.br
 * Version: 0.4.11
 * Created: 27 October 2026
 * Update Date: 27 October 2026
 *
 * Description:
 *     Port of vaila/freekiki_physics.py (simulate_flight, fit_flight) to the browser. The report
 *     embeds the per-frame DLT projections and the ball pixels of the run; the reader moves the
 *     physical anchors (metres on the pitch) and recomputes the fitted flight here. The model is
 *     the same: gravity, quadratic drag with the Brazuca Cd of Kiratidis & Leinweber, the Cd spin
 *     term and Magnus lift; the fit is a Levenberg-Marquardt over the same nine parameters, from
 *     the same start guesses. The integrator is a fixed-step RK4 (solve_ivp in Python).
 *
 *     The file has no dependency: it runs in the report HTML and, through module.exports, under
 *     node, where the Python tests compare it with the physics module.
 */
(function (root) {
  "use strict";

  var GRAVITY = 9.81;
  var AIR_DENSITY = 1.2;
  var AIR_VISCOSITY = 1.8e-5;
  var BALL_DIAMETER = 0.22;
  var BALL_MASS = 0.43;
  var BALL_RADIUS = BALL_DIAMETER / 2;
  var BALL_AREA = Math.PI * BALL_RADIUS * BALL_RADIUS;
  var K = (0.5 * AIR_DENSITY * BALL_AREA) / BALL_MASS;
  var SPIN_DRAG_B = 0.514;
  var SPIN_MAX_RPS = 12.0;
  var PIXEL_SIGMA = 2.0;
  var DEFAULT_SIGMA = 0.3;
  var STEP_S = 1 / 240; // RK4 step, seconds
  var BRAZUCA = {
    a: 0.474,
    v_c: 12.92,
    v_s: 1.0,
    b_min: 0.1657,
    b_max: 0.2112,
    v_min: 14.61,
    v_max: 35.0,
  };

  function clip(x, lo, hi) {
    return Math.min(Math.max(x, lo), hi);
  }

  function norm3(v) {
    return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }

  function cross3(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }

  function reynolds(speed) {
    return (speed * BALL_DIAMETER * AIR_DENSITY) / AIR_VISCOSITY;
  }

  function spinParameter(speed, omegaNorm) {
    return (BALL_RADIUS * omegaNorm) / Math.max(speed, 1e-6);
  }

  // Kiratidis Eq. 5 (sigmoid blend of the sub- and supercritical values) plus the spin term.
  function dragCoefficient(speed, sp) {
    var c = BRAZUCA;
    var v = Math.min(speed, c.v_max);
    var e1 = clip((v - c.v_c) / c.v_s, -60, 60);
    var e2 = clip(-(v - c.v_min) / c.v_s, -60, 60);
    var cd = (c.a - c.b_min) / (1 + Math.exp(e1)) + c.b_min;
    cd += ((v - c.v_min) / (1 + Math.exp(e2))) * (c.b_max - c.b_min) / (c.v_max - c.v_min);
    return cd + SPIN_DRAG_B * sp;
  }

  function liftCoefficient(sp) {
    return 1.15 * Math.pow(Math.max(sp, 0), 0.83);
  }

  // State s = [x, y, z, vx, vy, vz]; omega is the spin vector (rad/s).
  function derivative(s, omega, omegaNorm) {
    var vx = s[3];
    var vy = s[4];
    var vz = s[5];
    var v = [vx, vy, vz];
    var speed = norm3(v);
    if (speed < 1e-9) {
      return [vx, vy, vz, 0, 0, -GRAVITY];
    }
    var sp = spinParameter(speed, omegaNorm);
    var cd = dragCoefficient(speed, sp);
    var cl = liftCoefficient(sp);
    var spinDir = [0, 0, 0];
    if (omegaNorm > 0) {
      var wh = [omega[0] / omegaNorm, omega[1] / omegaNorm, omega[2] / omegaNorm];
      var vh = [vx / speed, vy / speed, vz / speed];
      spinDir = cross3(wh, vh);
    }
    var kd = K * cd * speed;
    var kl = K * cl * speed * speed;
    return [
      vx,
      vy,
      vz,
      -kd * vx + kl * spinDir[0],
      -kd * vy + kl * spinDir[1],
      -kd * vz + kl * spinDir[2] - GRAVITY,
    ];
  }

  function addScaled(s, k, h) {
    return [s[0] + h * k[0], s[1] + h * k[1], s[2] + h * k[2],
            s[3] + h * k[3], s[4] + h * k[4], s[5] + h * k[5]]; // prettier-ignore
  }

  function rk4(s, h, omega, omegaNorm) {
    var k1 = derivative(s, omega, omegaNorm);
    var k2 = derivative(addScaled(s, k1, h / 2), omega, omegaNorm);
    var k3 = derivative(addScaled(s, k2, h / 2), omega, omegaNorm);
    var k4 = derivative(addScaled(s, k3, h), omega, omegaNorm);
    var out = new Array(6);
    for (var i = 0; i < 6; i++) {
      out[i] = s[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }
    return out;
  }

  // Ball centre [x, y, z] at each time (ascending) for theta = [x0, y0, vx, vy, vz, tk, wx, wy, wz].
  // Before tk the ball rests at (x0, y0, restHeight).
  function simulate(theta, times, restHeight) {
    var x0 = theta[0];
    var y0 = theta[1];
    var tk = theta[5];
    var omega = [theta[6], theta[7], theta[8]];
    var omegaNorm = norm3(omega);
    var out = new Array(times.length);
    var s = [x0, y0, restHeight, theta[2], theta[3], theta[4]];
    var t = tk;
    for (var j = 0; j < times.length; j++) {
      if (times[j] < tk) {
        out[j] = [x0, y0, restHeight];
        continue;
      }
      while (t < times[j] - 1e-12) {
        var h = Math.min(STEP_S, times[j] - t);
        s = rk4(s, h, omega, omegaNorm);
        t += h;
      }
      out[j] = [s[0], s[1], s[2]];
    }
    return out;
  }

  // The time of every frame, f / fps (computed once per data set).
  function frameTimes(data) {
    if (!data.times) {
      data.times = [];
      for (var f = 0; f < data.n; f++) data.times.push(f / data.fps);
    }
    return data.times;
  }

  // Residuals in the order of fit_flight: pixels / PIXEL_SIGMA, then each anchor point / sigma.
  function residuals(theta, data) {
    var size = 2 * data.frames.length + 3 * data.points.length;
    var pos = simulate(theta, frameTimes(data), data.rest_height);
    for (var i = 0; i < pos.length; i++) {
      if (!isFinite(pos[i][0]) || !isFinite(pos[i][1]) || !isFinite(pos[i][2])) {
        return new Array(size).fill(1e3);
      }
    }
    var res = [];
    for (var f = 0; f < data.frames.length; f++) {
      var fr = data.frames[f];
      var p = pos[fr.frame];
      var P = fr.P;
      var h0 = P[0] * p[0] + P[1] * p[1] + P[2] * p[2] + P[3];
      var h1 = P[4] * p[0] + P[5] * p[1] + P[6] * p[2] + P[7];
      var h2 = P[8] * p[0] + P[9] * p[1] + P[10] * p[2] + P[11];
      var ok = h2 > 1e-6;
      var u = ok ? h0 / h2 : 1e4;
      var v = ok ? h1 / h2 : 1e4;
      res.push((u - fr.u) / PIXEL_SIGMA, (v - fr.v) / PIXEL_SIGMA);
    }
    for (var k = 0; k < data.points.length; k++) {
      var pt = data.points[k];
      var q = pos[pt.frame];
      for (var c = 0; c < 3; c++) {
        res.push((q[c] - pt.xyz[c]) / pt.sigma);
      }
    }
    return res;
  }

  function sumSquares(r) {
    var s = 0;
    for (var i = 0; i < r.length; i++) s += r[i] * r[i];
    return s;
  }

  // Solves the small dense system A x = b by Gaussian elimination with partial pivoting.
  function solveLinear(A, b) {
    var n = b.length;
    var M = A.map(function (row, i) {
      return row.concat([b[i]]);
    });
    for (var col = 0; col < n; col++) {
      var piv = col;
      for (var r = col + 1; r < n; r++) {
        if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      }
      var tmp = M[col];
      M[col] = M[piv];
      M[piv] = tmp;
      if (Math.abs(M[col][col]) < 1e-300) return null;
      for (var row = col + 1; row < n; row++) {
        var factor = M[row][col] / M[col][col];
        for (var k = col; k <= n; k++) M[row][k] -= factor * M[col][k];
      }
    }
    var x = new Array(n);
    for (var i = n - 1; i >= 0; i--) {
      var acc = M[i][n];
      for (var j = i + 1; j < n; j++) acc -= M[i][j] * x[j];
      x[i] = acc / M[i][i];
    }
    return x;
  }

  // Levenberg-Marquardt with a forward-difference Jacobian, clipped to the bounds.
  function levenberg(fun, x0, lo, hi, maxIter) {
    var p = x0.length;
    var x = x0.map(function (v, i) {
      return clip(v, lo[i], hi[i]);
    });
    var r = fun(x);
    var cost = sumSquares(r);
    var lambda = 1e-3;
    var stalls = 0;
    for (var iter = 0; iter < maxIter; iter++) {
      var J = [];
      for (var j = 0; j < p; j++) J.push(null);
      for (var jj = 0; jj < p; jj++) {
        var h = 1e-6 * Math.max(1, Math.abs(x[jj]));
        var xp = x.slice();
        xp[jj] = clip(x[jj] + h, lo[jj], hi[jj]);
        var step = xp[jj] - x[jj];
        if (step === 0) {
          xp[jj] = clip(x[jj] - h, lo[jj], hi[jj]);
          step = xp[jj] - x[jj];
        }
        var rp = fun(xp);
        J[jj] = rp.map(function (value, i) {
          return step === 0 ? 0 : (value - r[i]) / step;
        });
      }
      var A = [];
      var g = new Array(p).fill(0);
      for (var a = 0; a < p; a++) {
        A.push(new Array(p).fill(0));
        for (var b = 0; b < p; b++) {
          var s = 0;
          for (var i2 = 0; i2 < r.length; i2++) s += J[a][i2] * J[b][i2];
          A[a][b] = s;
        }
        var gs = 0;
        for (var i3 = 0; i3 < r.length; i3++) gs += J[a][i3] * r[i3];
        g[a] = gs;
      }
      var accepted = false;
      for (var trial = 0; trial < 25 && !accepted; trial++) {
        var M = A.map(function (row, i) {
          var copy = row.slice();
          copy[i] += lambda * (row[i] + 1e-12);
          return copy;
        });
        var delta = solveLinear(M, g.map(function (v) { return -v; }));
        if (delta === null) {
          lambda *= 10;
          continue;
        }
        var xn = x.map(function (v, i) {
          return clip(v + delta[i], lo[i], hi[i]);
        });
        var rn = fun(xn);
        var costn = sumSquares(rn);
        if (costn < cost) {
          var relative = (cost - costn) / Math.max(cost, 1e-300);
          x = xn;
          r = rn;
          cost = costn;
          lambda = Math.max(lambda / 3, 1e-9);
          accepted = true;
          stalls = relative < 1e-10 ? stalls + 1 : 0;
        } else {
          lambda *= 4;
        }
      }
      if (!accepted || stalls >= 3) break;
    }
    return { x: x, cost: cost, residuals: r };
  }

  // The fit of fit_flight: several kick times and spin starts, the best cost is kept.
  function fitFlight(data) {
    var fps = data.fps;
    var n = data.n;
    var points = data.points;
    var rest = data.rest_height;
    var earliest = points[0];
    var latest = points[0];
    for (var i = 1; i < points.length; i++) {
      if (points[i].frame < earliest.frame) earliest = points[i];
      if (points[i].frame > latest.frame) latest = points[i];
    }
    var x0g = earliest.xyz[0];
    var y0g = earliest.xyz[1];
    var target = latest.frame > 0 ? latest : null;
    var omegaMax = 2 * Math.PI * SPIN_MAX_RPS;
    var lo = [-80, -60, -80, -80, -80, 0, -omegaMax, -omegaMax, -omegaMax];
    var hi = [80, 60, 80, 80, 80, (n - 1) / fps, omegaMax, omegaMax, omegaMax];
    var spins = [[0, 0, 0], [0, 0, 2 * Math.PI * 6], [0, 0, -2 * Math.PI * 6]];
    var tks = [0, 0.5 / fps, 1 / fps];
    var best = null;
    for (var t = 0; t < tks.length; t++) {
      var v = [0, 0, 0];
      if (target) {
        var T = Math.max(target.frame / fps - tks[t], 1e-3);
        v = [
          (target.xyz[0] - x0g) / T,
          (target.xyz[1] - y0g) / T,
          (target.xyz[2] - rest + 0.5 * GRAVITY * T * T) / T,
        ];
      }
      for (var s = 0; s < spins.length; s++) {
        var guess = [x0g, y0g, v[0], v[1], v[2], tks[t], spins[s][0], spins[s][1], spins[s][2]];
        var sol = levenberg(
          function (th) {
            return residuals(th, data);
          },
          guess,
          lo,
          hi,
          60
        );
        if (best === null || sol.cost < best.cost) best = sol;
      }
    }
    return metrics(best.x, data, best.cost);
  }

  // Everything the report shows, computed from theta and the data (no fit).
  function metrics(theta, data, cost) {
    var n = data.n;
    var fps = data.fps;
    var pos = simulate(theta, frameTimes(data), data.rest_height);
    var res = residuals(theta, data);
    var pxSq = 0;
    var perFrame = {};
    var maxPx = 0;
    for (var f = 0; f < data.frames.length; f++) {
      var ex = res[2 * f] * PIXEL_SIGMA;
      var ey = res[2 * f + 1] * PIXEL_SIGMA;
      pxSq += ex * ex + ey * ey;
      var e = Math.hypot(ex, ey);
      perFrame[data.frames[f].frame] = e;
      if (e > maxPx) maxPx = e;
    }
    var rms = Math.sqrt(pxSq / (2 * data.frames.length));
    var anchorM = [];
    var worst = 0;
    for (var k = 0; k < data.points.length; k++) {
      var pt = data.points[k];
      var q = pos[pt.frame];
      var d = Math.hypot(q[0] - pt.xyz[0], q[1] - pt.xyz[1], q[2] - pt.xyz[2]);
      anchorM.push(d);
      if (d > worst) worst = d;
    }
    var speed = new Array(n).fill(0);
    var peak = 0;
    for (var i = 1; i < n; i++) {
      speed[i] = norm3([
        pos[i][0] - pos[i - 1][0],
        pos[i][1] - pos[i - 1][1],
        pos[i][2] - pos[i - 1][2],
      ]) * fps;
      if (speed[i] > peak) peak = speed[i];
    }
    var lastFrame = 0;
    for (var m = 0; m < data.frames.length; m++) {
      if (data.frames[m].frame > lastFrame) lastFrame = data.frames[m].frame;
    }
    var kickF = clip(Math.round(theta[5] * fps), 0, Math.max(lastFrame - 1, 0));
    var dx = pos[lastFrame][0] - pos[kickF][0];
    var dy = pos[lastFrame][1] - pos[kickF][1];
    var zmax = -Infinity;
    for (var z = 0; z < n; z++) {
      if (pos[z][2] > zmax) zmax = pos[z][2];
    }
    var omega = [theta[6], theta[7], theta[8]];
    var tier = worst < 0.3 ? "alta" : worst < 1.0 ? "média" : "baixa";
    return {
      theta: theta.slice(),
      cost: cost === undefined ? null : cost,
      positions: pos,
      rms_px: rms,
      max_px: maxPx,
      per_frame_px: perFrame,
      anchor_m: anchorM,
      worst_m: worst,
      tier: tier,
      kick_time_s: theta[5],
      spin_rps: norm3(omega) / (2 * Math.PI),
      speed_mps: speed,
      peak_kmh: peak * 3.6,
      distance_m: Math.hypot(dx, dy),
      zmax_m: zmax,
      kick_frame: kickF,
      end_frame: lastFrame,
    };
  }

  // ---------------------------------------------------------------------------
  // Browser: the editor section of the report
  // ---------------------------------------------------------------------------
  var api = {
    simulate: simulate,
    residuals: residuals,
    fitFlight: fitFlight,
    metrics: metrics,
    dragCoefficient: dragCoefficient,
    reynolds: reynolds,
    STEP_S: STEP_S,
  };

  function initEditor(doc) {
    var dataEl = doc.getElementById("ed-data");
    if (!dataEl) return;
    var data = JSON.parse(dataEl.textContent);
    var L = data.labels;
    var lang = data.lang;
    var state = { points: data.points.map(clonePoint), theta: data.theta.slice() };
    frameTimes(data);

    function num(x, nd) {
      if (!isFinite(x)) return "—";
      var s = x.toFixed(nd);
      return lang === "pt" ? s.replace(".", ",") : s;
    }

    function clonePoint(p) {
      return { label: p.label, frame: p.frame, xyz: p.xyz.slice(), sigma: p.sigma };
    }

    function $(id) {
      return doc.getElementById(id);
    }

    function renderRows() {
      var body = $("ed-rows");
      body.innerHTML = "";
      state.points.forEach(function (p, index) {
        var tr = doc.createElement("tr");
        tr.appendChild(textCell("text", p.label, "ed-label"));
        tr.appendChild(inputCell("number", p.frame, "ed-frame", 0));
        tr.appendChild(inputCell("number", p.xyz[0], "ed-x", 2));
        tr.appendChild(inputCell("number", p.xyz[1], "ed-y", 2));
        tr.appendChild(inputCell("number", p.xyz[2], "ed-z", 2));
        tr.appendChild(inputCell("number", p.sigma, "ed-sigma", 2));
        var cell = doc.createElement("td");
        var button = doc.createElement("button");
        button.type = "button";
        button.textContent = "×";
        button.setAttribute("aria-label", L.ed_col_remove);
        button.addEventListener("click", function () {
          state.points.splice(index, 1);
          renderRows();
        });
        cell.appendChild(button);
        tr.appendChild(cell);
        body.appendChild(tr);
      });
    }

    function textCell(type, value, cls) {
      var td = doc.createElement("td");
      td.className = cls;
      td.textContent = value;
      return td;
    }

    function inputCell(type, value, cls, step) {
      var td = doc.createElement("td");
      var input = doc.createElement("input");
      input.type = type;
      input.className = cls;
      input.value = value;
      input.step = step === 0 ? "1" : "0.01";
      td.appendChild(input);
      return td;
    }

    function readRows() {
      var rows = doc.querySelectorAll("#ed-rows tr");
      var points = [];
      rows.forEach(function (tr, index) {
        var inputs = tr.querySelectorAll("input");
        var label = tr.querySelector("td").textContent;
        var frame = Math.round(parseFloat(inputs[0].value));
        var xyz = [parseFloat(inputs[1].value), parseFloat(inputs[2].value), parseFloat(inputs[3].value)];
        var sigma = parseFloat(inputs[4].value);
        if (!isFinite(frame) || frame < 0 || frame >= data.n) {
          throw new Error("frame " + (index + 1) + " is outside 0.." + (data.n - 1));
        }
        if (xyz.some(function (c) { return !isFinite(c); }) || !(sigma > 0)) {
          throw new Error("point " + (index + 1) + ": x, y, z and sigma must be numbers (sigma > 0)");
        }
        points.push({ label: label, frame: frame, xyz: xyz, sigma: sigma });
      });
      return points;
    }

    function showMetrics(m, title) {
      var rows = [];
      function row(key, value) {
        rows.push("<dt>" + key + "</dt><dd>" + value + "</dd>");
      }
      var tierLabel = L["tier_" + m.tier] || m.tier;
      row(L.ed_out_tier, "<b>" + tierLabel + "</b>");
      row(L.ed_out_rms, num(m.rms_px, 2) + " px · " + L.ed_out_max.toLowerCase() + " " + num(m.max_px, 2) + " px");
      row(L.ed_out_anchor, m.anchor_m.map(function (d, i) {
        return "p" + (i + 1) + " " + num(d, 2) + " m";
      }).join(" · "));
      row(L.ed_out_kick, num(m.kick_time_s, 3) + " s");
      row(L.ed_out_spin, num(m.spin_rps, 1) + " rps");
      row(L.ed_out_peak, num(m.peak_kmh, 0) + " km/h");
      row(L.ed_out_dist, num(m.distance_m, 1) + " m");
      row(L.ed_out_zmax, num(m.zmax_m, 2) + " m");
      $("ed-out").innerHTML = rows.join("");
      $("ed-status").textContent = title;
      drawPlots(m);
    }

    function drawPlots(m) {
      drawTop($("ed-top"), m);
      drawSide($("ed-side"), m);
    }

    function canvasFrame(canvas, xs, ys) {
      var ctx = canvas.getContext("2d");
      var w = canvas.width;
      var h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      var minX = Math.min.apply(null, xs);
      var maxX = Math.max.apply(null, xs);
      var minY = Math.min.apply(null, ys);
      var maxY = Math.max.apply(null, ys);
      var pad = 0.5;
      minX -= pad; maxX += pad; minY -= pad; maxY += pad;
      var s = Math.min((w - 40) / (maxX - minX), (h - 40) / (maxY - minY));
      return {
        ctx: ctx,
        map: function (x, y) {
          return [20 + (x - minX) * s, h - 20 - (y - minY) * s];
        },
      };
    }

    function drawTop(canvas, m) {
      var k0 = m.kick_frame;
      var k1 = m.end_frame;
      var xy = [];
      for (var i = k0; i <= k1; i++) xy.push(m.positions[i]);
      var xs = xy.map(function (p) { return p[0]; });
      var ys = xy.map(function (p) { return p[1]; });
      var frame = canvasFrame(canvas, xs.concat(data.points.map(function (p) { return p.xyz[0]; })),
        ys.concat(data.points.map(function (p) { return p.xyz[1]; })));
      var ctx = frame.ctx;
      ctx.strokeStyle = "#d62246";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      xy.forEach(function (p, i) {
        var q = frame.map(p[0], p[1]);
        if (i === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
      });
      ctx.stroke();
      ctx.fillStyle = "#14213d";
      data.points.forEach(function (p, j) {
        var q = frame.map(p.xyz[0], p.xyz[1]);
        ctx.beginPath();
        ctx.arc(q[0], q[1], 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillText("p" + (j + 1), q[0] + 7, q[1] - 6);
      });
    }

    function drawSide(canvas, m) {
      var k0 = m.kick_frame;
      var k1 = m.end_frame;
      var ox = m.positions[k0][0];
      var oy = m.positions[k0][1];
      var rs = [];
      var zs = [];
      for (var i = k0; i <= k1; i++) {
        var p = m.positions[i];
        rs.push(Math.hypot(p[0] - ox, p[1] - oy));
        zs.push(p[2]);
      }
      var anchorR = data.points.map(function (p) {
        var q = m.positions[p.frame];
        return Math.hypot(p.xyz[0] - ox, p.xyz[1] - oy);
      });
      var frame = canvasFrame(canvas, rs.concat(anchorR), zs.concat(data.points.map(function (p) { return p.xyz[2]; })));
      var ctx = frame.ctx;
      ctx.strokeStyle = "#d62246";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      rs.forEach(function (r, i) {
        var q = frame.map(r, zs[i]);
        if (i === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
      });
      ctx.stroke();
      ctx.fillStyle = "#14213d";
      data.points.forEach(function (p, j) {
        var q = frame.map(anchorR[j], p.xyz[2]);
        ctx.beginPath();
        ctx.arc(q[0], q[1], 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillText("p" + (j + 1), q[0] + 7, q[1] - 6);
      });
    }

    function exportJson() {
      var out = {
        points: state.points.map(function (p) {
          return { frame: p.frame, xyz: p.xyz, sigma: p.sigma, label: p.label };
        }),
      };
      var text = JSON.stringify(out, null, 2);
      $("ed-json").value = text;
      return text;
    }

    function recalc() {
      $("ed-status").textContent = L.ed_running;
      setTimeout(function () {
        try {
          state.points = readRows();
          if (state.points.length < 3) {
            $("ed-status").textContent = L.ed_min;
            return;
          }
          data.points = state.points;
          var m = fitFlight(data);
          state.theta = m.theta;
          showMetrics(m, L.ed_done);
          exportJson();
        } catch (err) {
          $("ed-status").textContent = String(err.message || err);
        }
      }, 30);
    }

    $("ed-add").addEventListener("click", function () {
      state.points = readRowsSafe();
      state.points.push({ label: "ponto", frame: 0, xyz: [0, 0, 0], sigma: DEFAULT_SIGMA });
      renderRows();
    });
    $("ed-recalc").addEventListener("click", recalc);
    $("ed-reset").addEventListener("click", function () {
      state.points = data.original_points.map(clonePoint);
      data.points = state.points;
      renderRows();
      showMetrics(metrics(data.theta, data, null), L.ed_done);
      exportJson();
    });
    $("ed-download").addEventListener("click", function () {
      var text = exportJson();
      var blob = new Blob([text], { type: "application/json" });
      var link = doc.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "ball_anchors.json";
      link.click();
    });

    function readRowsSafe() {
      try {
        return readRows();
      } catch (err) {
        return state.points;
      }
    }

    $("ed-title-text").textContent = L.ed_title;
    renderRows();
    data.points = state.points;
    showMetrics(metrics(data.theta, data, null), L.ed_done);
    exportJson();
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  if (typeof document !== "undefined" && root && typeof root.addEventListener === "function") {
    root.addEventListener("DOMContentLoaded", function () {
      initEditor(document);
    });
  }
})(typeof window !== "undefined" ? window : undefined);
