OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[0];
z q[1];
tdg q[1];
s q[1];
t q[0];
id q[1];
u1(4.766355421295945) q[1];
id q[1];
t q[1];
s q[0];
sdg q[1];
id q[1];
t q[1];
rz(5.78719793864599) q[0];
phase(3.5267637737227098) q[1];
s q[1];
tdg q[1];
sdg q[1];
rz(4.857888023972658) q[1];
z q[1];
sdg q[1];
phase(0.16937237097237628) q[1];
u1(1.899208992403034) q[1];
sdg q[0];
rz(3.0915450189974125) q[1];
z q[1];
rz(0.8167055141794105) q[1];
z q[0];
z q[0];
sdg q[0];
tdg q[0];
tdg q[0];
z q[0];
Rzz(4.290109354777065) q[0], q[1];
Rzz(4.763349598514234) q[0], q[0];
Rzz(3.958609240178365) q[0], q[1];
sdg q[0];
s q[0];
phase(3.864814759264955) q[1];
id q[0];
s q[1];
rz(2.2271841280428144) q[1];
tdg q[0];
id q[1];
phase(3.703954120573372) q[0];
u1(0.7427974709855725) q[1];
z q[1];
id q[1];
phase(2.668866272588034) q[1];
z q[0];
t q[0];
u1(1.7231921091251174) q[0];
sdg q[0];
t q[0];
sdg q[0];
s q[1];
u1(4.567601798804703) q[0];
Rzz(0.5554797092416383) q[1], q[0];
tdg q[1];
phase(0.11183765116744174) q[1];
id q[0];
s q[0];
tdg q[0];
z q[1];
rz(3.2309594534427326) q[0];
tdg q[0];
t q[1];
u1(4.495631535348441) q[1];
id q[1];
tdg q[1];
id q[1];
Rzz(3.4943143079175454) q[0], q[1];
sdg q[1];
Rzz(0.8303194492871572) q[0], q[1];
u1(3.5467074791368884) q[0];
u1(1.6008848304381167) q[1];
phase(0.24386733622275988) q[1];
id q[0];
id q[0];
sdg q[1];
id q[0];
