OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rz(5.923721493105617) q[0];
rz(3.5830559075635486) q[1];
rz(4.018561776166972) q[0];
rz(0.13156331827792925) q[1];
rz(5.156726018327123) q[0];
rz(3.3968667855282657) q[1];
rz(4.433626177876975) q[1];
rz(3.2641324987825597) q[0];
rz(4.504433610741894) q[1];
rz(3.172078247592733) q[0];
cx q[0], q[1];
rz(1.3082815403982968) q[1];
rz(0.2334261145976982) q[0];
rz(5.405255007364069) q[1];
rz(0.03734143175874598) q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
rz(4.5645750294747875) q[0];
rz(0.6305585635262498) q[1];
rz(0.11595424395784734) q[1];
rz(0.743254837945751) q[0];
rz(3.335012987609028) q[1];
rz(1.6662163423707772) q[0];
rz(3.820785025256279) q[0];
rz(4.281146171538772) q[1];
cx q[0], q[1];
rz(2.4098344898334494) q[1];
rz(5.587484270858506) q[0];
rz(0.4362810771464325) q[0];
rz(6.245838841173107) q[1];