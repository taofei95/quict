OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
sdg q[4];
t q[1];
rz(5.575277340746064) q[3];
id q[5];
s q[2];
x q[5];
t q[5];
rx(2.083209244345481) q[5];
tdg q[3];
cu1(2.96477724030323) q[5], q[1];
rxx(4.453561893548714) q[1], q[0];
u2(6.23580686512248, 4.919292466151763) q[4];
rx(4.769728623481473) q[4];
u2(0.46325544835800125, 1.9216796302648251) q[3];
u3(3.948349572000424, 3.3819051552397377, 2.61090456418539) q[4];
u3(3.337164663456838, 5.629762415150578, 5.11806613647421) q[2];
rzz(1.461043775702213) q[0], q[2];
u3(2.6117599951243156, 2.7702499726700056, 2.6653952798581138) q[2];
u3(4.512469374478211, 0.33248915445813676, 1.7903627696722564) q[2];
h q[4];
rx(2.440495105120878) q[1];
t q[4];
id q[3];
tdg q[4];
rz(4.133321406270246) q[0];
tdg q[0];
x q[1];
u3(2.2395778293790314, 5.280638009904906, 5.519852979299668) q[1];
id q[4];
ch q[5], q[0];