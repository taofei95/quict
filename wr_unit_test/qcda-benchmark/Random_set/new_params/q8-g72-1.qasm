OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
s q[2];
id q[2];
swap q[1], q[7];
rx(0.08726379767781443) q[0];
id q[1];
p(2.7826271340418116) q[2];
t q[7];
u2(2.8730307181301167, 0.7920165826685873) q[1];
rz(4.7692904913124865) q[6];
p(3.6648436958992727) q[3];
cy q[0], q[5];
ryy(5.452205214422113) q[5], q[4];
crz(6.260040058209227) q[2], q[1];
tdg q[1];
u3(3.8121986257654648, 1.522542142945659, 0.8873438011391861) q[5];
h q[1];
u1(2.708343474155461) q[2];
p(3.971771781527765) q[7];
t q[0];
tdg q[6];
t q[2];
id q[0];
t q[2];
ch q[2], q[7];
cx q[2], q[1];
p(4.306682726992695) q[5];
s q[6];
rx(5.165087271402062) q[0];
crz(1.1939330530656544) q[1], q[5];
rz(4.362755276104545) q[3];
rzz(4.014894069600311) q[0], q[1];
t q[4];
x q[4];
ry(5.719419585459761) q[3];
s q[0];
ch q[5], q[3];
s q[7];
cy q[7], q[0];
rx(1.9436782062485425) q[1];
x q[0];
t q[7];
u2(4.719285302300403, 5.943031220878288) q[3];
id q[5];
p(5.884513600446522) q[1];
sdg q[0];
ryy(1.7224633469806274) q[6], q[3];
x q[1];
p(0.6397377620474178) q[0];
tdg q[2];
crz(1.0632660000206384) q[2], q[0];
sdg q[1];
rx(0.8742536869015344) q[4];
rx(4.7804924222334835) q[5];
rz(1.4035184647794159) q[2];
u2(2.257578876614004, 3.4648672570613166) q[6];
t q[5];
ry(5.26609174205342) q[6];
tdg q[3];
t q[3];
rxx(1.7081745444376861) q[1], q[3];
id q[6];
u1(4.298631411555646) q[1];
rx(5.799145303770527) q[1];
p(3.4120149939037345) q[2];
h q[1];
rz(1.494785295928505) q[6];
ryy(4.962429595631065) q[4], q[5];
cu1(1.8319534502394554) q[4], q[5];
sdg q[5];
tdg q[2];
h q[1];
id q[4];