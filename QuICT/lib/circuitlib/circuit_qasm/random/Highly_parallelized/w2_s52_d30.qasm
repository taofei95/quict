OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
cx q[1], q[0];
rz(5.983742212782628) q[1];
rz(1.0561798395495374) q[0];
rz(1.8220254215653249) q[0];
rz(3.365419145504979) q[1];
rz(0.8725898978938705) q[0];
rz(2.2171651424119325) q[1];
cx q[0], q[1];
rz(0.6860919582369929) q[0];
rz(0.9353513494640765) q[1];
cx q[0], q[1];
rz(1.6353580244735626) q[1];
rz(3.5214717382086107) q[0];
cx q[1], q[0];
rz(1.2446988907314753) q[0];
rz(2.8633824337957563) q[1];
rz(4.3323514787739095) q[0];
rz(3.000772766824483) q[1];
rz(3.3375932616725534) q[0];
rz(6.190323430043072) q[1];
rz(0.8010940488608563) q[1];
rz(5.440945616269999) q[0];
cx q[1], q[0];
rz(5.188198469661174) q[1];
rz(4.596734124217335) q[0];
rz(4.261139480851546) q[0];
rz(3.9809733559697476) q[1];
rz(2.459726030675666) q[0];
rz(0.3151181752128616) q[1];
cx q[0], q[1];
cx q[0], q[1];
rz(3.973313031926547) q[0];
rz(3.5112778630140076) q[1];
rz(0.0036126109794403565) q[0];
rz(3.8658770329493715) q[1];
rz(6.040466552280515) q[0];
rz(4.561840789966459) q[1];
cx q[0], q[1];
rz(2.6945673200887725) q[1];
rz(0.2071211476300793) q[0];
rz(2.0199467838086003) q[0];
rz(4.547384996661253) q[1];
rz(3.246687695583562) q[1];
rz(1.0392437990023349) q[0];
rz(3.490651218888186) q[0];
rz(5.036750772106369) q[1];
rz(0.510826626319387) q[0];
rz(3.417316562178876) q[1];
rz(1.381234615443129) q[1];
rz(3.316377368472092) q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];