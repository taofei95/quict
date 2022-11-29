<template>
  <div>
    <div id="l_drawZone"></div>
    
  </div>
</template>
<style>
.el-dialog {
  --el-dialog-background-color: #13141c;
  fill: white;
}
.el-dialog__title {
  --el-color-text-primary: white;
}
</style>
<script>
import * as d3 from "d3";

export default {
  props: ["VisContentIn"],
  data: function () {
    return {
      VisContent: undefined,
      drawZone: undefined,
      controlTargetZone: undefined,
      paramsZone: undefined,
      gateEdit: {
        name: "",
      },
    };
  },
  methods: {
    vis_change() { // gate图变化，重新绘制
      // TODO: copy viscontentin
      this.VisContent = JSON.parse(JSON.stringify(this.VisContentIn));
      let maxPosX = 0;
      this.VisContent.gates.forEach((e) => {
        let max = 0;
        let min = this.VisContent.q.length;
        if (e.posX > maxPosX) {
          maxPosX = e.posX;
        }
        e.controls_delta = [];
        e.controls.forEach((c) => {
          e.controls_delta.push(c - e.q);
          if (c - e.q > max) {
            max = c - e.q;
          }
          if (c - e.q < min) {
            min = c - e.q;
          }
        });
        e.targets_delta = [];
        e.targets.forEach((t) => {
          e.targets_delta.push({ delta: t - e.q, img: e.img });
          if (t - e.q > max) {
            max = t - e.q;
          }
          if (t - e.q < min) {
            min = t - e.q;
          }
        });
        e.line_start = min;
        e.line_end = max;
      });

      let width = 1000;
      let gateSize = 40;
      let maxWidth = maxPosX * gateSize + 300;
      if (maxWidth > width) {
        width = maxWidth;
      }
      let height = 500;
      let maxHeight = this.VisContent.q.length * gateSize + 50;
      if (maxHeight > height) {
        height = maxHeight;
      }

      if (this.drawZone != undefined) {
        d3.select("#l_drawZone").selectAll("*").remove();
      }
      this.drawZone = d3
        .select("#l_drawZone")
        .append("svg")
        .attr("width", `${width*1.25}px`)
        .attr("height", `${height*1.25}px`)
        .attr("viewBox", "0,0," + width + "," + height)
        .attr("background", "blue")
        .append("g")
        .style("transform", () => {
          return `translateX(100px) translateY(30px)`;
        });


      // draw q
      let q = this.drawZone
        .selectAll(".qNode")
        .data(this.VisContent.q)
        .join("g")
        .classed("qNode", true)
        .style("transform", (d, i) => {
          return `translateX(10px) translateY(${i * gateSize + 20}px)`;
        });
      q.append("line")
        .attr("stroke", "#3D4054")
        .attr("stroke-width", 1.5)
        .attr("fill", "#3D4054")

        .attr("x1", -10)
        .attr("y1", 10)
        .attr("x2", width - 200)
        .attr("y2", 10);

      q.append("text")
        .text((d) => `Q [${d}]`)
        .style("fill", "white")
        .style("transform", () => {
          return `translateX(-50px) translateY(${gateSize / 2 - 5}px)`;
        });


      // draw gates
      let gates = this.drawZone
        .selectAll(".gatesNode")
        .data(this.VisContent.gates)
        .join("g")
        .classed("gatesNode", true)
        .style("transform", (d) => {
          return `translateX(${d.posX * gateSize + 10 + 10}px) translateY(${
            d.q * gateSize + 20 + 10
          }px)`;
        });
      // gate_line
      gates
        .append("line")
        .attr("stroke-width", 2)
        .attr("stroke", "green")
        .attr("x1", 0)
        .attr("y1", (d) => d.line_start * gateSize)
        .attr("x2", 0)
        .attr("y2", (d) => d.line_end * gateSize);

      // control_nodes
      gates
        .selectAll(".controlNode")
        .data((d) => {
          return d.controls_delta;
        })
        .join("circle")
        .classed("controlNode", true)
        .attr("stroke", "steelblue")
        .attr("stroke-width", 1.5)
        .attr("fill", "lightblue")
        .attr("r", 5)
        .style("transform", (d) => {
          return `translateX(0px) translateY(${d * gateSize}px)`;
        });

      // target_nodes
      gates
        .selectAll(".targetNode")
        .data((d) => {
          return d.targets_delta;
        })
        .join("image")
        .classed("targetNode", true)
        .attr("width", gateSize - 2)
        .attr("height", gateSize - 2)
        .attr("xlink:href", (d) => `./assets/gate/${d.img}`)
        .style("transform", (d) => {
          return `translateX(-${gateSize / 2}px) translateY(${
            d.delta * gateSize - gateSize / 2
          }px)`;
        });

    },
  },
  mounted: function () {
    this.vis_change();
  },
  watch: {},
  emits: {
  },
};
</script>