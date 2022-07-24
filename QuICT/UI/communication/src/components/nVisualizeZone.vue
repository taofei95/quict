<template>
  <div>
    <div id="n_drawZone"></div>
    <el-dialog
      :title="'Edit ' + gateEdit.name"
      v-model="dialogGateVisible"
      width="30%"
    >
      <div id="paramsZone"></div>
      <div id="controlTargetZone">
        <svg></svg>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogGateVisible = false">取 消</el-button>
          <el-button type="primary" @click="gate_edited">确 定</el-button>
        </span>
      </template>
    </el-dialog>
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
      dialogGateVisible: false,
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

      let thisRef = this;
      let width = 1000;
      let gateSize = 40;
      let maxWidth = maxPosX * gateSize + 300;
      if (maxWidth > width) {
        width = maxWidth;
      }
      let height = 500;
      let maxHeight = this.VisContent.q.length * gateSize + 200;
      if (maxHeight > height) {
        height = maxHeight;
      }

      if (this.drawZone != undefined) {
        d3.select("#n_drawZone").selectAll("*").remove();
      }
      this.drawZone = d3
        .select("#n_drawZone")
        .append("svg")
        .attr("width", `${width*1.25}px`)
        .attr("height", `${height*1.25}px`)
        .attr("viewBox", "0,0," + width + "," + height)
        .attr("background", "blue")
        .append("g")
        .style("transform", () => {
          return `translateX(100px) translateY(30px)`;
        });

      // draw gate set
      let gateSet = this.drawZone
        .selectAll(".gateSetNode")
        .data(this.VisContent.gateSet)
        .join("g")
        .classed("gateSetNode", true)
        .style("transform", (d, i) => {
          return `translateX(${(i % 20) * gateSize}px) translateY(${
            Math.floor(i / 20) * gateSize + 10
          }px)`;
        });

      // gateSet icon

      gateSet
        .append("image")
        .attr("width", gateSize - 2)
        .attr("height", gateSize - 2)
        .attr("xlink:href", (d) => `./assets/gate_set/${d.img}`);
      gateSet.append("title").text((d) => d.matrix);

      gateSet.call(
        d3
          .drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)
      );

      function dragstarted() {
        d3.select(this).classed("dragging", true);
      }

      function dragged(event) {
        d3.select(this).style("transform", () => {
          return `translateX(${event.x}px) translateY(${event.y}px)`;
        });
      }

      function dragended(event, d) { // 从instruction Set 中拖拽gate到gate列表上添加新gate
        d3.select(this).classed("dragging", false);
        let xi = Math.round((event.x - 10) / gateSize);
        let yi = Math.round((event.y - 100) / gateSize);
        console.log(event, xi, yi);
        if (xi >= 0 && yi >= 0) {
          thisRef.$emit("VisUpdate", {
            type: "gates add",
            x: xi,
            y: yi,
            gate: d,
          });
        }
      }

      // draw q
      let q = this.drawZone
        .selectAll(".qNode")
        .data(this.VisContent.q)
        .join("g")
        .classed("qNode", true)
        .style("transform", (d, i) => {
          return `translateX(10px) translateY(${i * gateSize + 100}px)`;
        });
      q.append("line")
        .attr("stroke", "#3D4054")
        .attr("stroke-width", 1.5)
        .attr("fill", "#3D4054")

        .attr("x1", -10)
        .attr("y1", 10)
        .attr("x2", width - 200)
        .attr("y2", 10);

      let q_head = q
        .append("image")
        .attr("width", gateSize - 2)
        .attr("height", gateSize - 2)
        .attr("xlink:href", `./assets/delete.2x.png`)
        .style("transform", () => {
          return `translateX(-90px) translateY(-10px)`;
        });

      q.append("text")
        .text((d) => `Q [${d}]`)
        .style("fill", "white")
        .style("transform", () => {
          return `translateX(-50px) translateY(${gateSize / 2 - 5}px)`;
        });

      q_head.on("click", (event, d) => { // 通知外层删除qbit
        console.log(event, d);
        thisRef.$emit("VisUpdate", { type: "q remove", index: d });
      });

      // draw gates
      let gates = this.drawZone
        .selectAll(".gatesNode")
        .data(this.VisContent.gates)
        .join("g")
        .classed("gatesNode", true)
        .style("transform", (d) => {
          return `translateX(${d.posX * gateSize + 10 + 10}px) translateY(${
            d.q * gateSize + 100 + 10
          }px)`;
        })
        .call(
          d3
            .drag()
            .on("start", dragstarted_gates)
            .on("drag", dragged_gates)
            .on("end", dragended_gates)
        )
        .on("mouseenter", gate_enter)
        .on("mouseleave", gate_leave);
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

      // gate ctrls
      let gate_edit = gates
        .append("g")
        .classed("gate-ctrl", true)
        .style("background-color", "#D4D6DA")
        .style("transform", () => {
          return `translateX(-20px) translateY(-20px)`;
        })
        .style("display", "none")
        .on("click", function (event, d) { // 显示gate编辑界面
          console.log(event, d);
          // copy d to gate edit
          thisRef.gateEdit = JSON.parse(JSON.stringify(d));
          thisRef.dialogGateVisible = true;
          setTimeout(updateDialogGate, 500);
        });

      gate_edit
        .append("image")
        .attr("xlink:href", `./assets/edit.png`)
        .attr("width", 12)
        .attr("height", 12);

      let gate_remove = gates
        .append("g")
        .classed("gate-ctrl", true)
        .style("background-color", "#D4D6DA")
        .style("transform", () => {
          return `translateX(7px) translateY(-20px)`;
        })
        .style("display", "none")
        .on("click", function (event, d) { // 通知外层删除gate
          console.log(event, d);
          thisRef.$emit("VisUpdate", { type: "gates remove", index: d.index });
        });

      gate_remove
        .append("image")
        .attr("xlink:href", `./assets/remove.png`)
        .attr("width", 12)
        .attr("height", 12);

      function dragstarted_gates(event) {
        console.log(event);
      }

      function dragged_gates(event) {
        console.log(event);
        d3.select(this).classed("dragging", true);

        d3.select(this).style("transform", () => {
          return `translateX(${event.x}px) translateY(${event.y}px)`;
        });
      }

      function dragended_gates(event, d) { // 拖拽gate到gate列表其他位置
        console.log(event);
        if (d3.select(this).classed("dragging") == true) {
          d3.select(this).classed("dragging", false);
          // d3.select(this)
          // .selectAll(".gate-ctrl")
          // .style("display", "block");
          let xi = Math.round((event.x - 10) / gateSize);
          let yi = Math.round((event.y - 100) / gateSize);
          if (xi >= 0 && yi >= 0) {
            thisRef.$emit("VisUpdate", {
              type: "gates move",
              x: xi,
              y: yi,
              gate: d,
            });
          }
        }
      }

      function gate_enter(event, d) { //显示编辑/删除gate按钮
        console.log(event);
        if (event.defaultPrevented) return;
        d3.select(this).selectAll(".gate-ctrl").style("display", "block");

        d.selected = true;
        //
      }

      function gate_leave(event, d) { //隐藏编辑/删除gate按钮
        console.log(event);
        if (event.defaultPrevented) return;
        d.selected = false;
        setTimeout(() => {
          if (d.selected) {
            return;
          }
          d3.select(this).selectAll(".gate-ctrl").style("display", "none");
        }, 500);

        //
      }

      function updateDialogGate() { // 绘制gate编辑界面
        //draw params zone
        if (thisRef.paramsZone != undefined) {
          d3.select("#paramsZone").selectAll("*").remove();
        }
        thisRef.paramsZone = d3.select("#paramsZone");

        let params = thisRef.paramsZone
          .selectAll(".paramsInput")
          .data(thisRef.gateEdit.pargs)
          .join("div")
          .classed("paramsInput", true);
        params
          .append("span")
          .append("b")
          .text((d, i) => `param [${i}]`)
          .style("margin", "10px");
        params
          .append("input")
          .attr("value", (d) => d)
          .attr("idx", (d, i) => i)
          .style("margin", "10px")
          .on("change", (e) => {
            thisRef.gateEdit.pargs[Number(d3.select(e.target).attr("idx"))] =
              e.target.value;
          });

        let dialogHeight = 250;
        let dialogHeight_max = thisRef.VisContent.q.length * 30 + 100;
        if (dialogHeight_max > dialogHeight) {
          dialogHeight = dialogHeight_max;
        }

        //draw control-target zone
        if (thisRef.controlTargetZone != undefined) {
          d3.select("#controlTargetZone").select("svg").selectAll("*").remove();
        }
        thisRef.controlTargetZone = d3
          .select("#controlTargetZone")
          .select("svg")
          .attr("width", "200px")
          .attr("height", `${dialogHeight}px`)
          .attr("background", "blue")
          .append("g")
          .style("transform", () => {
            return `translateX(70px) translateY(30px)`;
          });

        thisRef.gateEdit.qConnected = [];
        thisRef.VisContent.q.forEach(() => {
          thisRef.gateEdit.qConnected.push(-1);
        });

        // draw q connected
        let q_root = thisRef.controlTargetZone
          .selectAll(".qNode")
          .data(thisRef.gateEdit.qConnected)
          .join("g")
          .classed("qNode", true)
          .style("transform", (d, i) => {
            return `translateX(10px) translateY(${i * 30}px)`;
          });
        // draw q
        q_root
          .append("circle")
          .attr("stroke", "steelblue")
          .attr("stroke-width", 1.5)
          .attr("fill", "lightblue")
          .attr("cx", -10)
          // .attr("cy", 10)
          .attr("r", 10);

        q_root
          .append("text")
          .text((d, i) => `Q [${i}]`)
          .style("transform", () => {
            return `translateX(-60px) translateY(5px)`;
          });

        thisRef.gateEdit.target_control = [];

        for (
          let i = 0;
          i <
          thisRef.gateEdit.targets.length + thisRef.gateEdit.controls.length;
          i++
        ) {
          if (i >= thisRef.gateEdit.controls.length) {
            // push targets point
            let j = i - thisRef.gateEdit.controls.length;
            thisRef.gateEdit.target_control.push({
              index: i,
              type: "target",
              connectedTo: thisRef.gateEdit.targets[j],
            });
            thisRef.gateEdit.qConnected[thisRef.gateEdit.targets[j]] = i;
          } else {
            // push target point
            thisRef.gateEdit.target_control.push({
              index: i,
              type: "control",
              connectedTo: thisRef.gateEdit.controls[i],
            });
            thisRef.gateEdit.qConnected[thisRef.gateEdit.controls[i]] = i;
          }
        }

        let target_control_root = thisRef.controlTargetZone
          .selectAll(".ctNode")
          .data(thisRef.gateEdit.target_control)
          .join("g")
          .classed("ctNode", true)
          .style("transform", (d, i) => {
            return `translateX(100px) translateY(${i * 30}px)`;
          });

        //draw connect lines
        thisRef.controlTargetZone
          .selectAll(".qConnect")
          .data(thisRef.gateEdit.qConnected)
          .join("line")
          .classed("qConnect", true)
          .attr("stroke", (d) => {
            if (d == -1) {
              return "transparent";
            } else {
              return "steelblue";
            }
          })
          .attr("stroke-width", 1.5)
          .attr("x1", 10)
          .attr("y1", (d, i) => {
            return `${i * 30}px`;
          })
          .attr("x2", 80)
          .attr("y2", (d) => {
            return `${d * 30}px`;
          });

        // draw target_control
        target_control_root
          .append("circle")
          .attr("stroke", "steelblue")
          .attr("stroke-width", 1.5)
          .attr("fill", "lightblue")
          .attr("cx", -15)
          // .attr("cy", 10)
          .attr("r", 5)
          .style("display", (d) => {
            if (d.type == "control") {
              return "block";
            } else {
              return "none";
            }
          });

        target_control_root
          .append("image")
          .attr("width", 28)
          .attr("height", 28)
          .attr("xlink:href", () => `./assets/gate/${thisRef.gateEdit.img}`)
          .style("transform", () => {
            return `translateX(-30px) translateY(-14px)`;
          })
          .style("display", (d) => {
            if (d.type == "target") {
              return "block";
            } else {
              return "none";
            }
          });

        target_control_root.call(
          d3
            .drag()
            .on("start", dragstarted_connect)
            .on("drag", dragged_connect)
            .on("end", dragended_connect)
        );

        function dragstarted_connect() {
          d3.select(this).classed("dragging", true);
        }

        function dragged_connect(event, d) {
          thisRef.controlTargetZone.selectAll(".draggedConnect").remove();
          thisRef.controlTargetZone
            .append("g")
            .classed("draggedConnect", true)
            .append("line")
            .attr("stroke", "darkblue")
            .attr("stroke-width", 1.5)
            .attr("x1", 80)
            .attr("y1", () => {
              return `${d.index * 30}px`;
            })
            .attr("x2", () => {
              return event.x;
            })
            .attr("y2", () => {
              return event.y;
            });
        }

        function dragended_connect(event, d) { // 拖拽当前control/target到qbit修改作用位
          d3.select(this).classed("dragging", false);
          thisRef.controlTargetZone.selectAll(".draggedConnect").remove();
          let xi = Math.round((event.x - 10) / 30);
          let yi = Math.round(event.y / 30);
          console.log(event, xi, yi);
          if (
            xi >= 0 &&
            xi <= 1 &&
            yi >= 0 &&
            yi < thisRef.VisContent.q.length
          ) {
            let conflicted = false;
            thisRef.gateEdit.target_control.forEach((el) => {
              if (el.connectedTo == yi) {
                conflicted = true;
              }
            });
            if (!conflicted) {
              if (d.index >= thisRef.gateEdit.controls.length) {
                // is targets point
                let j = d.index - thisRef.gateEdit.controls.length;
                thisRef.gateEdit.targets[j] = yi;
              } else {
                // is target point
                thisRef.gateEdit.controls[d.index] = yi;
              }
              updateDialogGate();
            }
          }
        }
      }

      // draw btns
      let q_add = this.drawZone
        .append("g")
        .classed("q_add", true)
        .style("transform", () => {
          return `translateX(10px) translateY(${
            this.VisContent.q.length * gateSize + 100 + 10
          }px)`;
        });
      q_add
        .append("image")
        .attr("width", gateSize - 2)
        .attr("height", gateSize - 2)
        .attr("xlink:href", `./assets/add.2x.png`)
        .style("transform", () => {
          return `translateX(-90px) translateY(-5px)`;
        });

      q_add
        .append("text")
        .text("Add")
        .style("fill", "white")
        .style("transform", () => {
          return `translateX(-50px) translateY(${gateSize / 2}px)`;
        });

      q_add.on("click", () => { // 通知外层新增qbit
        thisRef.$emit("VisUpdate", { type: "q add" });
      });
    },
    gate_edited() { // 通知外层编辑gate
      this.dialogGateVisible = false;
      this.$emit("VisUpdate", { type: "gates edit", gate: this.gateEdit });
    },
  },
  mounted: function () {
    this.vis_change();
  },
  watch: {},
  emits: {
    VisUpdate: null,
  },
};
</script>