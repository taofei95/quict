OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
sqiswap q[5], q[6];
rz(-2.9604561285444713) q[5];
rz(6.102048782134265) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[4], q[5];
rz(-4.940993180340284) q[4];
rz(8.082585833930077) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[6], q[7];
rz(-3.153082066656435) q[6];
rz(6.294674720246228) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
sqiswap q[3], q[4];
rz(-0.6566943531233759) q[3];
rz(3.798287006713169) q[4];
sqiswap q[3], q[4];
rz(-3.141592653589793) q[4];
sqiswap q[5], q[6];
rz(-0.13215052679870626) q[5];
rz(3.2737431803884993) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[7], q[8];
rz(-0.2647667048408441) q[7];
rz(3.4063593584306373) q[8];
sqiswap q[7], q[8];
rz(-3.141592653589793) q[8];
sqiswap q[2], q[3];
rz(-5.4977506760653965) q[2];
rz(8.63934332965519) q[3];
sqiswap q[2], q[3];
rz(-3.141592653589793) q[3];
sqiswap q[4], q[5];
rz(-1.4231081558232848) q[4];
rz(4.564700809413078) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[6], q[7];
rz(-1.862952047435193) q[6];
rz(5.004544701024987) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
sqiswap q[8], q[9];
rz(-4.105511254471621) q[8];
rz(7.247103908061414) q[9];
sqiswap q[8], q[9];
rz(-3.141592653589793) q[9];
sqiswap q[1], q[2];
rz(-3.0527346635755066) q[1];
rz(6.1943273171653) q[2];
sqiswap q[1], q[2];
rz(-3.141592653589793) q[2];
sqiswap q[3], q[4];
rz(-4.834937601766572) q[3];
rz(7.976530255356365) q[4];
sqiswap q[3], q[4];
rz(-3.141592653589793) q[4];
sqiswap q[5], q[6];
rz(-5.297934114427628) q[5];
rz(8.439526768017421) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[7], q[8];
rz(-0.5295122397521427) q[7];
rz(3.6711048933419357) q[8];
sqiswap q[7], q[8];
rz(-3.141592653589793) q[8];
sqiswap q[9], q[10];
rz(-2.631469208648013) q[9];
rz(5.773061862237807) q[10];
sqiswap q[9], q[10];
rz(-3.141592653589793) q[10];
sqiswap q[0], q[1];
rz(-5.536862085044766) q[0];
rz(8.67845473863456) q[1];
sqiswap q[0], q[1];
rz(-3.141592653589793) q[1];
sqiswap q[2], q[3];
rz(-5.75714165965849) q[2];
rz(8.898734313248283) q[3];
sqiswap q[2], q[3];
rz(-3.141592653589793) q[3];
sqiswap q[4], q[5];
rz(-2.689575756153719) q[4];
rz(5.831168409743512) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[6], q[7];
rz(-0.8975389238554663) q[6];
rz(4.039131577445259) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
sqiswap q[8], q[9];
rz(-4.471613883666523) q[8];
rz(7.613206537256316) q[9];
sqiswap q[8], q[9];
rz(-3.141592653589793) q[9];
sqiswap q[10], q[11];
rz(-4.410051875344812) q[10];
rz(7.551644528934605) q[11];
sqiswap q[10], q[11];
rz(-3.141592653589793) q[11];
sqiswap q[1], q[2];
rz(-2.2379309544977515) q[1];
rz(5.379523608087545) q[2];
sqiswap q[1], q[2];
rz(-3.141592653589793) q[2];
sqiswap q[3], q[4];
rz(-4.214296683409975) q[3];
rz(7.355889336999768) q[4];
sqiswap q[3], q[4];
rz(-3.141592653589793) q[4];
sqiswap q[5], q[6];
rz(-1.7061625983523363) q[5];
rz(4.84775525194213) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[7], q[8];
rz(-2.1712812292417856) q[7];
rz(5.312873882831578) q[8];
sqiswap q[7], q[8];
rz(-3.141592653589793) q[8];
sqiswap q[9], q[10];
rz(-5.837465220732449) q[9];
rz(8.979057874322242) q[10];
sqiswap q[9], q[10];
rz(-3.141592653589793) q[10];
sqiswap q[11], q[12];
rz(-5.421003554727917) q[11];
rz(8.56259620831771) q[12];
sqiswap q[11], q[12];
rz(-3.141592653589793) q[12];
sqiswap q[2], q[3];
rz(-0.6726730515424093) q[2];
rz(3.814265705132202) q[3];
sqiswap q[2], q[3];
rz(-3.141592653589793) q[3];
sqiswap q[4], q[5];
rz(-1.256029321555968) q[4];
rz(4.397621975145761) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[6], q[7];
rz(-5.85275076776702) q[6];
rz(8.994343421356813) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
sqiswap q[8], q[9];
rz(-2.681730770907845) q[8];
rz(5.823323424497638) q[9];
sqiswap q[8], q[9];
rz(-3.141592653589793) q[9];
sqiswap q[10], q[11];
rz(-5.903776049406654) q[10];
rz(9.045368702996447) q[11];
sqiswap q[10], q[11];
rz(-3.141592653589793) q[11];
sqiswap q[3], q[4];
rz(-1.2840184566945136) q[3];
rz(4.425611110284307) q[4];
sqiswap q[3], q[4];
rz(-3.141592653589793) q[4];
sqiswap q[5], q[6];
rz(-4.568146705336999) q[5];
rz(7.709739358926792) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[7], q[8];
rz(-1.470503849446639) q[7];
rz(4.612096503036432) q[8];
sqiswap q[7], q[8];
rz(-3.141592653589793) q[8];
sqiswap q[9], q[10];
rz(-3.9916828476418287) q[9];
rz(7.133275501231622) q[10];
sqiswap q[9], q[10];
rz(-3.141592653589793) q[10];
sqiswap q[4], q[5];
rz(-5.538676144598607) q[4];
rz(8.6802687981884) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[6], q[7];
rz(-4.6791439658111305) q[6];
rz(7.820736619400924) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
sqiswap q[8], q[9];
rz(-0.8474748297089958) q[8];
rz(3.989067483298789) q[9];
sqiswap q[8], q[9];
rz(-3.141592653589793) q[9];
sqiswap q[5], q[6];
rz(-5.322749209932589) q[5];
rz(8.464341863522382) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[7], q[8];
rz(-4.110795536939704) q[7];
rz(7.252388190529497) q[8];
sqiswap q[7], q[8];
rz(-3.141592653589793) q[8];
sqiswap q[6], q[7];
rz(-3.855782494777766) q[6];
rz(6.997375148367559) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
