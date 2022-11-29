OPENQASM 3.0;
include "stdgates.qasm";

qubit qs[7];

x qs[0];
x qs[1];
x qs[2];
// iSWAP Gate
cx qs[1], qs[0];
rz(-4.033799032958379) qs[1];
ry(-6.95298287684501) qs[1];
cx qs[0], qs[1];
ry(6.95298287684501) qs[1];
rz(4.033799032958379) qs[1];
cx qs[1], qs[0];
// iSWAP Gate
cx qs[3], qs[2];
rz(-5.3479135187270055) qs[3];
ry(-7.593801602882474) qs[3];
cx qs[2], qs[3];
ry(7.593801602882474) qs[3];
rz(5.3479135187270055) qs[3];
cx qs[3], qs[2];
// iSWAP Gate
cx qs[5], qs[4];
rz(-8.585846880530141) qs[5];
ry(-3.5138227213643685) qs[5];
cx qs[4], qs[5];
ry(3.5138227213643685) qs[5];
rz(8.585846880530141) qs[5];
cx qs[5], qs[4];
// iSWAP Gate
cx qs[2], qs[1];
rz(-3.38584629065516) qs[2];
ry(-6.457163522373925) qs[2];
cx qs[1], qs[2];
ry(6.457163522373925) qs[2];
rz(3.38584629065516) qs[2];
cx qs[2], qs[1];
// iSWAP Gate
cx qs[4], qs[3];
rz(-6.588820282482613) qs[4];
ry(-4.1567074984266075) qs[4];
cx qs[3], qs[4];
ry(4.1567074984266075) qs[4];
rz(6.588820282482613) qs[4];
cx qs[4], qs[3];
// iSWAP Gate
cx qs[6], qs[5];
rz(-5.088422678370297) qs[6];
ry(-6.186484128925519) qs[6];
cx qs[5], qs[6];
ry(6.186484128925519) qs[6];
rz(5.088422678370297) qs[6];
cx qs[6], qs[5];
// iSWAP Gate
cx qs[1], qs[0];
rz(-8.815742555225128) qs[1];
ry(-2.364005995294423) qs[1];
cx qs[0], qs[1];
ry(2.364005995294423) qs[1];
rz(8.815742555225128) qs[1];
cx qs[1], qs[0];
// iSWAP Gate
cx qs[3], qs[2];
rz(-3.553081985806257) qs[3];
ry(-2.359090247886406) qs[3];
cx qs[2], qs[3];
ry(2.359090247886406) qs[3];
rz(3.553081985806257) qs[3];
cx qs[3], qs[2];
// iSWAP Gate
cx qs[5], qs[4];
rz(-6.4332640662474425) qs[5];
ry(-3.2683881134750967) qs[5];
cx qs[4], qs[5];
ry(3.2683881134750967) qs[5];
rz(6.4332640662474425) qs[5];
cx qs[5], qs[4];
// iSWAP Gate
cx qs[2], qs[1];
rz(-3.665452727734224) qs[2];
ry(-2.3674447642659753) qs[2];
cx qs[1], qs[2];
ry(2.3674447642659753) qs[2];
rz(3.665452727734224) qs[2];
cx qs[2], qs[1];
// iSWAP Gate
cx qs[4], qs[3];
rz(-6.44899823869398) qs[4];
ry(-3.5782086646462314) qs[4];
cx qs[3], qs[4];
ry(3.5782086646462314) qs[4];
rz(6.44899823869398) qs[4];
cx qs[4], qs[3];
// iSWAP Gate
cx qs[6], qs[5];
rz(-3.9068260719962313) qs[6];
ry(-6.146560978571178) qs[6];
cx qs[5], qs[6];
ry(6.146560978571178) qs[6];
rz(3.9068260719962313) qs[6];
cx qs[6], qs[5];
// iSWAP Gate
cx qs[1], qs[0];
rz(-6.765445138295984) qs[1];
ry(-3.5135066365927012) qs[1];
cx qs[0], qs[1];
ry(3.5135066365927012) qs[1];
rz(6.765445138295984) qs[1];
cx qs[1], qs[0];
// iSWAP Gate
cx qs[3], qs[2];
rz(-5.3629404980152495) qs[3];
ry(-3.4351822940935417) qs[3];
cx qs[2], qs[3];
ry(3.4351822940935417) qs[3];
rz(5.3629404980152495) qs[3];
cx qs[3], qs[2];
// iSWAP Gate
cx qs[5], qs[4];
rz(-4.393229783942838) qs[5];
ry(-6.942007336765906) qs[5];
cx qs[4], qs[5];
ry(6.942007336765906) qs[5];
rz(4.393229783942838) qs[5];
cx qs[5], qs[4];
// iSWAP Gate
cx qs[2], qs[1];
rz(-7.160120004836034) qs[2];
ry(-7.184036723174281) qs[2];
cx qs[1], qs[2];
ry(7.184036723174281) qs[2];
rz(7.160120004836034) qs[2];
cx qs[2], qs[1];
// iSWAP Gate
cx qs[4], qs[3];
rz(-3.965472026669337) qs[4];
ry(-1.868614831246528) qs[4];
cx qs[3], qs[4];
ry(1.868614831246528) qs[4];
rz(3.965472026669337) qs[4];
cx qs[4], qs[3];
// iSWAP Gate
cx qs[6], qs[5];
rz(-6.399250413218757) qs[6];
ry(-3.780687373687895) qs[6];
cx qs[5], qs[6];
ry(3.780687373687895) qs[6];
rz(6.399250413218757) qs[6];
cx qs[6], qs[5];