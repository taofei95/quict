OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
creg c[14];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
x q[6];
sqiswap q[6], q[7];
rz(-3.131521522157537) q[6];
rz(6.27311417574733) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
sqiswap q[5], q[6];
rz(-5.327905532227724) q[5];
rz(8.469498185817518) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[7], q[8];
rz(-3.785189996543502) q[7];
rz(6.926782650133295) q[8];
sqiswap q[7], q[8];
rz(-3.141592653589793) q[8];
sqiswap q[4], q[5];
rz(-1.4340310655015773) q[4];
rz(4.575623719091371) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[6], q[7];
rz(-1.745827988638431) q[6];
rz(4.887420642228224) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
sqiswap q[8], q[9];
rz(-5.420652824284121) q[8];
rz(8.562245477873915) q[9];
sqiswap q[8], q[9];
rz(-3.141592653589793) q[9];
sqiswap q[3], q[4];
rz(-4.983474640046855) q[3];
rz(8.125067293636647) q[4];
sqiswap q[3], q[4];
rz(-3.141592653589793) q[4];
sqiswap q[5], q[6];
rz(-2.948074273083779) q[5];
rz(6.089666926673572) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[7], q[8];
rz(-4.673310434729559) q[7];
rz(7.814903088319352) q[8];
sqiswap q[7], q[8];
rz(-3.141592653589793) q[8];
sqiswap q[9], q[10];
rz(-2.9244998288324178) q[9];
rz(6.066092482422211) q[10];
sqiswap q[9], q[10];
rz(-3.141592653589793) q[10];
sqiswap q[2], q[3];
rz(-5.196440539785625) q[2];
rz(8.338033193375418) q[3];
sqiswap q[2], q[3];
rz(-3.141592653589793) q[3];
sqiswap q[4], q[5];
rz(-3.001453390803064) q[4];
rz(6.143046044392857) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[6], q[7];
rz(-3.651256900379623) q[6];
rz(6.7928495539694165) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
sqiswap q[8], q[9];
rz(-1.6896496765698186) q[8];
rz(4.831242330159611) q[9];
sqiswap q[8], q[9];
rz(-3.141592653589793) q[9];
sqiswap q[10], q[11];
rz(-5.567306510233362) q[10];
rz(8.708899163823155) q[11];
sqiswap q[10], q[11];
rz(-3.141592653589793) q[11];
sqiswap q[1], q[2];
rz(-2.143933465935057) q[1];
rz(5.2855261195248495) q[2];
sqiswap q[1], q[2];
rz(-3.141592653589793) q[2];
sqiswap q[3], q[4];
rz(-2.064000434525174) q[3];
rz(5.205593088114967) q[4];
sqiswap q[3], q[4];
rz(-3.141592653589793) q[4];
sqiswap q[5], q[6];
rz(-0.47211332178646676) q[5];
rz(3.6137059753762597) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[7], q[8];
rz(-3.2400819623368844) q[7];
rz(6.381674615926677) q[8];
sqiswap q[7], q[8];
rz(-3.141592653589793) q[8];
sqiswap q[9], q[10];
rz(-3.0815294036812966) q[9];
rz(6.223122057271089) q[10];
sqiswap q[9], q[10];
rz(-3.141592653589793) q[10];
sqiswap q[11], q[12];
rz(-0.33452083231956065) q[11];
rz(3.476113485909354) q[12];
sqiswap q[11], q[12];
rz(-3.141592653589793) q[12];
sqiswap q[0], q[1];
rz(-3.127729715929607) q[0];
rz(6.269322369519401) q[1];
sqiswap q[0], q[1];
rz(-3.141592653589793) q[1];
sqiswap q[2], q[3];
rz(-2.992127990264936) q[2];
rz(6.133720643854729) q[3];
sqiswap q[2], q[3];
rz(-3.141592653589793) q[3];
sqiswap q[4], q[5];
rz(-5.654275889830751) q[4];
rz(8.795868543420543) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[6], q[7];
rz(-4.161676960015431) q[6];
rz(7.303269613605224) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
sqiswap q[8], q[9];
rz(-3.7187499479381008) q[8];
rz(6.860342601527893) q[9];
sqiswap q[8], q[9];
rz(-3.141592653589793) q[9];
sqiswap q[10], q[11];
rz(-4.700649131475854) q[10];
rz(7.842241785065647) q[11];
sqiswap q[10], q[11];
rz(-3.141592653589793) q[11];
sqiswap q[12], q[13];
rz(-0.6266219394876271) q[12];
rz(3.7682145930774205) q[13];
sqiswap q[12], q[13];
rz(-3.141592653589793) q[13];
sqiswap q[1], q[2];
rz(-1.7215765290502556) q[1];
rz(4.863169182640049) q[2];
sqiswap q[1], q[2];
rz(-3.141592653589793) q[2];
sqiswap q[3], q[4];
rz(-3.62208883143371) q[3];
rz(6.763681485023503) q[4];
sqiswap q[3], q[4];
rz(-3.141592653589793) q[4];
sqiswap q[5], q[6];
rz(-3.0992257346153163) q[5];
rz(6.24081838820511) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[7], q[8];
rz(-4.055959034832974) q[7];
rz(7.1975516884227675) q[8];
sqiswap q[7], q[8];
rz(-3.141592653589793) q[8];
sqiswap q[9], q[10];
rz(-1.6577206265923252) q[9];
rz(4.799313280182118) q[10];
sqiswap q[9], q[10];
rz(-3.141592653589793) q[10];
sqiswap q[11], q[12];
rz(-3.047521579885921) q[11];
rz(6.189114233475714) q[12];
sqiswap q[11], q[12];
rz(-3.141592653589793) q[12];
sqiswap q[2], q[3];
rz(-1.3615310492643047) q[2];
rz(4.503123702854098) q[3];
sqiswap q[2], q[3];
rz(-3.141592653589793) q[3];
sqiswap q[4], q[5];
rz(-1.176398064146059) q[4];
rz(4.317990717735852) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[6], q[7];
rz(-4.597595376135011) q[6];
rz(7.739188029724804) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
sqiswap q[8], q[9];
rz(-5.407670879011498) q[8];
rz(8.549263532601291) q[9];
sqiswap q[8], q[9];
rz(-3.141592653589793) q[9];
sqiswap q[10], q[11];
rz(-5.97485064082452) q[10];
rz(9.116443294414314) q[11];
sqiswap q[10], q[11];
rz(-3.141592653589793) q[11];
sqiswap q[3], q[4];
rz(-1.867180136674422) q[3];
rz(5.008772790264215) q[4];
sqiswap q[3], q[4];
rz(-3.141592653589793) q[4];
sqiswap q[5], q[6];
rz(-1.8042548224608648) q[5];
rz(4.945847476050658) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[7], q[8];
rz(-4.035455191026632) q[7];
rz(7.177047844616425) q[8];
sqiswap q[7], q[8];
rz(-3.141592653589793) q[8];
sqiswap q[9], q[10];
rz(-6.181708982377842) q[9];
rz(9.323301635967635) q[10];
sqiswap q[9], q[10];
rz(-3.141592653589793) q[10];
sqiswap q[4], q[5];
rz(-1.5028263595274445) q[4];
rz(4.644419013117238) q[5];
sqiswap q[4], q[5];
rz(-3.141592653589793) q[5];
sqiswap q[6], q[7];
rz(-4.64532115981407) q[6];
rz(7.786913813403863) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
sqiswap q[8], q[9];
rz(-1.1832931270297735) q[8];
rz(4.324885780619566) q[9];
sqiswap q[8], q[9];
rz(-3.141592653589793) q[9];
sqiswap q[5], q[6];
rz(-2.5111988835987664) q[5];
rz(5.652791537188559) q[6];
sqiswap q[5], q[6];
rz(-3.141592653589793) q[6];
sqiswap q[7], q[8];
rz(-5.799788837918029) q[7];
rz(8.941381491507823) q[8];
sqiswap q[7], q[8];
rz(-3.141592653589793) q[8];
sqiswap q[6], q[7];
rz(-4.013679589023511) q[6];
rz(7.155272242613304) q[7];
sqiswap q[6], q[7];
rz(-3.141592653589793) q[7];
