OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
x q[0];
x q[1];
x q[2];
x q[3];
x q[0];
h q[0];
rxx(0.4685010313987732) q[0], q[4];
rxx(0.8202890753746033) q[1], q[4];
rxx(0.20853477716445923) q[2], q[4];
rxx(0.01681685447692871) q[3], q[4];
ryy(0.08277994394302368) q[0], q[4];
ryy(0.7477876543998718) q[1], q[4];
ryy(0.9046382308006287) q[2], q[4];
ryy(0.4501489996910095) q[3], q[4];
rzx(0.49808239936828613) q[0], q[4];
rzx(0.6089378595352173) q[1], q[4];
rzx(0.9133774638175964) q[2], q[4];
rzx(0.15805143117904663) q[3], q[4];
h q[0];
