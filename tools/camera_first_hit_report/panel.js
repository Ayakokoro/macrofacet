(() => {
  "use strict";
  const root = document.getElementById("interactive-report");
  const data = JSON.parse(document.getElementById("ir-data").textContent);
  const byId = new Map(data.methods.map(m => [m.id, m]));
  const cases = new Map(data.cases.map(c => [c.id, c]));
  const windows = {early: "早碰撞", middle: "中段碰撞", late: "晚碰撞", all: "全部命中"};
  const selected = new Set(["local", "global_conditional", "renewal_0.1"]);
  const get = id => document.getElementById(id);
  const state = {baseline: "local", component: "W", scope: "windows", ray: data.rays[0].id, window: "middle"};
  const svgNS = "http://www.w3.org/2000/svg";
  let sequence = 0;
  const num = (x, places = 4) => Math.abs(x) < 5e-9 ? "0" : Number(x.toFixed(places)).toString();
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const svgEl = (tag, attrs = {}, text) => {
    const node = document.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const swatch = m => {
    const node = el("i", "ir-swatch");
    node.style.setProperty("--curve-color", m.color);
    if (m.dash !== "-") node.style.borderTopStyle = m.dash === ":" ? "dotted" : "dashed";
    return node;
  };
  const active = c => data.methods.filter(m => selected.has(m.id) && c.curves[m.id]);
  function bestPair(c) {
    return c.pairs.filter(p => p.sufficient && selected.has(p.method_a) && selected.has(p.method_b))
      .sort((a,b) => b.gap-a.gap)[0] || null;
  }
  function nearest(xs, x) {
    let lo = 0, hi = xs.length-1;
    while (lo < hi) {
      const mid = (lo+hi) >>> 1;
      if (xs[mid] < x) lo = mid+1; else hi = mid;
    }
    return lo > 0 && x-xs[lo-1] < xs[lo]-x ? lo-1 : lo;
  }

  class Chart {
    constructor(parent, c, difference, pair, onRange) {
      this.c = c; this.difference = difference; this.pair = pair; this.onRange = onRange;
      this.range = [c.x[0], c.x[c.x.length-1]];
      this.uid = `ir-clip-${sequence++}`;
      this.box = el("div", "ir-chart");
      this.svg = svgEl("svg", {class:"ir-svg", viewBox:`0 0 660 ${difference ? 200 : 325}`,
        role:"img", "aria-label":difference ? "相对于基准的曲线差值" : "方法分布曲线",
        "data-chart-kind":difference ? "difference" : "distribution"});
      this.tip = el("div", "ir-tooltip");
      this.box.append(this.svg,this.tip); parent.append(this.box);
      this.drag = null;
      this.svg.addEventListener("pointermove", e => this.move(e));
      this.svg.addEventListener("pointerleave", () => { if (!this.drag) this.hide(); });
      this.svg.addEventListener("pointerdown", e => {
        if (e.button !== 0) return;
        const p = this.pointer(e);
        if (p.x < 66 || p.x > 642 || p.y < 30 || p.y > this.bottom) return;
        this.drag = p.x;
        this.svg.setPointerCapture(e.pointerId);
        this.hide();
      });
      this.svg.addEventListener("pointerup", e => {
        if (this.drag === null) return;
        const a = this.drag, b = Math.max(66,Math.min(642,this.pointer(e).x));
        this.drag = null;
        if (this.svg.hasPointerCapture(e.pointerId)) this.svg.releasePointerCapture(e.pointerId);
        this.selection.setAttribute("visibility","hidden");
        if (Math.abs(a-b) > 8) this.onRange([this.unX(Math.min(a,b)),this.unX(Math.max(a,b))]);
      });
      this.svg.addEventListener("pointercancel", () => { this.drag=null; this.selection.setAttribute("visibility","hidden"); });
      this.svg.addEventListener("dblclick", () => this.onRange([c.x[0],c.x[c.x.length-1]]));
      this.draw();
    }
    pointer(e) {
      const p = this.svg.createSVGPoint(); p.x=e.clientX; p.y=e.clientY;
      return p.matrixTransform(this.svg.getScreenCTM().inverse());
    }
    unX(px) { return this.range[0]+(px-66)/576*(this.range[1]-this.range[0]); }
    setRange(range) { this.range=range; this.draw(); }
    value(method, i) {
      const y = this.c.curves[method].y[i];
      return this.difference ? 100*(y-this.c.curves[state.baseline].y[i]) : y;
    }
    draw() {
      this.svg.replaceChildren(); this.hide();
      const c=this.c, methods=active(c), bottom=this.difference ? 151 : 276;
      this.bottom=bottom;
      this.svg.dataset.rangeMin=this.range[0]; this.svg.dataset.rangeMax=this.range[1];
      const baseOk = !this.difference || c.curves[state.baseline];
      let ymin=0, ymax=1;
      if (this.difference && baseOk) {
        let extent=0;
        for (const m of methods) for (let i=0;i<c.x.length;i++) {
          if (c.x[i] >= this.range[0] && c.x[i] <= this.range[1]) extent=Math.max(extent,Math.abs(this.value(m.id,i)));
        }
        ymax=Math.max(1,extent*1.15); ymin=-ymax;
      }
      this.toX=x => 66+(x-this.range[0])/(this.range[1]-this.range[0])*576;
      this.toY=y => bottom-(y-ymin)/(ymax-ymin)*(bottom-30);
      const label=this.difference ? `Δ${c.kind === "transmittance" ? "T" : "CDF"} / 百分点（相对 ${byId.get(state.baseline).label}）`
        : (c.kind === "transmittance" ? "透射率 T(s)" : "梯度 CDF");
      this.svg.append(svgEl("text",{x:66,y:18,"font-weight":600},label));
      const defs=svgEl("defs"),clip=svgEl("clipPath",{id:this.uid});
      clip.append(svgEl("rect",{x:66,y:30,width:576,height:bottom-30})); defs.append(clip);this.svg.append(defs);
      for (let k=0;k<=4;k++) {
        const yy=ymin+(ymax-ymin)*k/4,py=this.toY(yy);
        this.svg.append(svgEl("line",{x1:66,x2:642,y1:py,y2:py,stroke:Math.abs(yy)<1e-9 ? "#afbfce" : "#e8eef4"}));
        this.svg.append(svgEl("text",{x:58,y:py+4,"text-anchor":"end"},num(yy,this.difference ? 2 : 2)));
      }
      for (let k=0;k<=4;k++) {
        const xx=this.range[0]+(this.range[1]-this.range[0])*k/4,px=this.toX(xx);
        this.svg.append(svgEl("line",{x1:px,x2:px,y1:30,y2:bottom,stroke:"#eef2f6"}));
        this.svg.append(svgEl("text",{x:px,y:bottom+20,"text-anchor":"middle"},num(xx)));
      }
      this.svg.append(svgEl("text",{x:354,y:bottom+43,"text-anchor":"middle"},
        c.kind === "transmittance" ? "s · 从介质入口起算的世界距离" : data.components[c.component]));
      const lines=svgEl("g",{"clip-path":`url(#${this.uid})`});
      if (baseOk) for (const m of methods) {
        const points=c.x.map((x,i)=>`${this.toX(x).toFixed(3)},${this.toY(this.value(m.id,i)).toFixed(3)}`).join(" ");
        const dash={"-":"", "--":"8 5", ":":"2 4", "-.":"9 4 2 4"}[m.dash];
        lines.append(svgEl("polyline",{points,fill:"none",stroke:m.color,"stroke-width":2.2,
          "stroke-dasharray":dash,"data-method":m.id,"vector-effect":"non-scaling-stroke"}));
      }
      this.svg.append(lines);
      if (this.pair && this.pair.x >= this.range[0] && this.pair.x <= this.range[1]) {
        const px=this.toX(this.pair.x);
        this.svg.append(svgEl("line",{x1:px,x2:px,y1:30,y2:bottom,stroke:"#849bad","stroke-dasharray":"3 5"}));
        if (!this.difference) for (const key of ["a","b"]) {
          this.svg.append(svgEl("circle",{cx:px,cy:this.toY(this.pair[key]),r:4,fill:byId.get(this.pair[`method_${key}`]).color,stroke:"white","stroke-width":1.3}));
        }
      }
      if (!methods.length || !baseOk) this.svg.append(svgEl("text",{x:354,y:(30+bottom)/2,"text-anchor":"middle"},
        !methods.length ? "请勾选要显示的曲线" : "当前基准在此窗口没有有效样本"));
      this.cross=svgEl("line",{y1:30,y2:bottom,stroke:"#67839a","stroke-dasharray":"3 3",visibility:"hidden"});
      this.selection=svgEl("rect",{y:30,height:bottom-30,fill:"#297bb82b",stroke:"#297bb8",visibility:"hidden"});
      this.svg.append(this.cross,this.selection);
    }
    hide() { this.tip.style.display="none"; if (this.cross) this.cross.setAttribute("visibility","hidden"); }
    move(e) {
      const p=this.pointer(e),px=Math.max(66,Math.min(642,p.x));
      if (this.drag !== null) {
        this.selection.setAttribute("x",Math.min(px,this.drag));
        this.selection.setAttribute("width",Math.abs(px-this.drag));
        this.selection.setAttribute("visibility","visible"); return;
      }
      if (p.x<66 || p.x>642 || p.y<30 || p.y>this.bottom || !selected.size ||
          (this.difference && !this.c.curves[state.baseline])) { this.hide(); return; }
      const i=nearest(this.c.x,this.unX(px));
      this.cross.setAttribute("x1",this.toX(this.c.x[i]));this.cross.setAttribute("x2",this.toX(this.c.x[i]));
      this.cross.setAttribute("visibility","visible");this.tip.replaceChildren();
      this.tip.append(el("b","",`显示点 x = ${num(this.c.x[i],6)}`));
      for (const m of active(this.c)) {
        const row=el("div");row.append(swatch(m),el("span","",m.label),el("b","",`${num(this.value(m.id,i),5)}${this.difference ? " pp" : ""}`));this.tip.append(row);
      }
      this.tip.style.display="block";
      const rect=this.box.getBoundingClientRect();
      this.tip.style.left=`${Math.max(0,Math.min(e.clientX-rect.left+12,rect.width-this.tip.offsetWidth))}px`;
      this.tip.style.top=`${Math.max(0,e.clientY-rect.top-this.tip.offsetHeight-12)}px`;
    }
  }

  function card(c, pair, rank) {
    const item=el("article","ir-card");item.dataset.case=c.id;
    if (pair) { item.dataset.gap=pair.gap; item.dataset.pair=`${pair.method_a}|${pair.method_b}`; }
    const title=`${rank ? `#${rank} · ` : ""}射线 ${c.ray} · 像素 (${c.pixel.join(", ")})`;
    item.append(el("div","ir-card-title",title));
    if (pair) {
      const score=el("div","ir-card-score",(100*pair.gap).toFixed(2));
      score.append(el("small","",c.kind === "transmittance" ? "百分点 · max |ΔT|" : "百分点 · CDF 最大差"));item.append(score);
      item.append(el("p","ir-card-note",`${byId.get(pair.method_a).label} ↔ ${byId.get(pair.method_b).label}，最大差在 x=${num(pair.x,6)}`));
    }
    if (c.kind === "gradient") item.append(el("p","ir-card-note",c.bounds
      ? `${windows[c.window]}：s ∈ [${num(c.bounds[0],6)}, ${num(c.bounds[1],6)})`
      : "所有接受的首次碰撞：此分布也混合了碰撞距离的差异"));
    const charts=[];
    const updateRange=r => charts.forEach(chart=>chart.setRange(r));
    charts.push(new Chart(item,c,false,pair,updateRange),new Chart(item,c,true,pair,updateRange));
    const legend=el("div","ir-legend");
    for (const m of active(c)) { const l=el("span");l.append(swatch(m),el("span","",m.label));legend.append(l); }
    item.append(legend);
    if (c.kind === "gradient") item.append(el("p","ir-counts",active(c).map(m=>`${m.label}: ${c.curves[m.id].n.toLocaleString()} 次命中`).join("；")));
    const actions=el("div","ir-card-actions");
    if (pair) {
      const zoom=el("button","","放大最大差附近");zoom.type="button";
      zoom.addEventListener("click",()=>{
        const lo=c.x[0], hi=c.x[c.x.length-1],span=(hi-lo)*(c.kind === "transmittance" ? .20 : .35);
        const start=Math.max(lo,Math.min(hi-span,pair.x-span/2));updateRange([start,start+span]);
      });actions.append(zoom);
    }
    const reset=el("button","","恢复全范围");reset.type="button";
    reset.addEventListener("click",()=>updateRange([c.x[0],c.x[c.x.length-1]]));actions.append(reset);
    if (rank) {
      const inspect=el("button","","逐条查看这条射线");inspect.type="button";
      inspect.addEventListener("click",()=>{
        state.ray=c.ray;if(c.kind === "gradient")state.window=c.window;
        get("ir-ray").value=state.ray;get("ir-window").value=state.window;
        renderExplorer();get("ir-explore-t").scrollIntoView({behavior:"smooth",block:"start"});
      });actions.append(inspect);
    }
    item.append(actions);return item;
  }
  function renderRanking(kind) {
    const target=get(kind === "transmittance" ? "ir-top-transmittance" : "ir-top-gradient");target.replaceChildren();
    const candidates=data.cases.filter(c=>c.kind === kind && (kind !== "gradient" ||
      c.component === state.component && (state.scope === "all" ? c.window === "all" : c.window !== "all")));
    const ranked=candidates.map(c=>({c,pair:bestPair(c)})).filter(row=>row.pair).sort((a,b)=>b.pair.gap-a.pair.gap);
    if (!ranked.length) target.append(el("p","ir-empty","至少勾选两种 local、global_conditional 或 plus 方法，且该窗口每组至少有 512 次命中，才能比较排名。可在下方逐条查看曲线。"));
    for (let i=0;i<Math.min(3,ranked.length);i++) target.append(card(ranked[i].c,ranked[i].pair,i+1));
  }
  function renderExplorer() {
    for (const [id,cid] of [["ir-explore-t",`t-${state.ray}`],["ir-explore-g",`g-${state.ray}-${state.window}-${state.component}`]]) {
      const c=cases.get(cid);get(id).replaceChildren(card(c,bestPair(c),null));
    }
  }
  function render() {
    get("ir-selection-count").textContent=`已选 ${selected.size} / ${data.methods.length}`;
    get("ir-gradient-caption").textContent=`${data.components[state.component]} · ${state.scope === "windows" ? "早 / 中 / 晚窗口分别排名" : "按所有接受命中排名"}`;
    renderRanking("transmittance");renderRanking("gradient");renderExplorer();
    root.dataset.ready="true";root.dataset.selected=[...selected].join(",");
  }
  function options(select,items,current) {
    for (const [value,label] of items) {const o=el("option","",label);o.value=value;select.append(o);}
    select.value=current;
  }
  for (const m of data.methods) {
    const label=el("label","ir-method"),input=el("input");
    input.type="checkbox";input.value=m.id;input.checked=selected.has(m.id);input.id=`ir-show-${m.id}`;
    input.addEventListener("change",()=>{if(input.checked)selected.add(m.id);else selected.delete(m.id);render();});
    label.append(input,swatch(m),el("span","",m.label));get("ir-methods").append(label);
  }
  options(get("ir-baseline"),data.methods.map(m=>[m.id,m.label]),state.baseline);
  options(get("ir-component"),Object.entries(data.components),state.component);
  options(get("ir-ray"),data.rays.map(r=>[r.id,`射线 ${r.id} · (${r.pixel.join(", ")})`]),state.ray);
  for (const [id,key] of [["ir-baseline","baseline"],["ir-component","component"],["ir-scope","scope"],["ir-ray","ray"],["ir-window","window"]]) {
    get(id).addEventListener("change",e=>{state[key]=key === "ray" ? Number(e.target.value) : e.target.value;render();});
  }
  root.querySelectorAll("[data-preset]").forEach(button=>button.addEventListener("click",()=>{
    const preset=button.dataset.preset;
    const ids=preset === "all" ? data.methods.map(m=>m.id) : preset === "none" ? [] : preset === "plus"
      ? data.methods.filter(m=>m.ranked).map(m=>m.id)
      : ["local","global_conditional",preset === "plus001" ? "renewal_0.01" : "renewal_0.1"];
    selected.clear();ids.forEach(id=>selected.add(id));
    get("ir-methods").querySelectorAll("input").forEach(input=>{input.checked=selected.has(input.value);});render();
  }));
  render();
})();
