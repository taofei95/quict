OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rz(2.3556236137103985) q[7];
rz(0.4811428081673324) q[8];
rz(3.3378204775439473) q[6];
rz(1.4779766831833938) q[5];
rz(2.5756819592527926) q[2];
rz(3.873954526497906) q[4];
cx q[1], q[3];
rz(0.474133067965605) q[9];
rz(0.6195769593555124) q[0];
rz(1.7386575791126828) q[0];
rz(0.2486742367760235) q[6];
rz(5.5322010467916245) q[3];
rz(5.4381584984247) q[8];
rz(0.07683340867821893) q[9];
cx q[5], q[4];
cx q[7], q[2];
rz(0.5387871772678027) q[1];
rz(2.99340514684804) q[1];
rz(2.117459122216821) q[0];
cx q[2], q[3];
rz(0.7879240756895683) q[9];
rz(3.347282131687224) q[7];
rz(0.707373782687811) q[6];
cx q[8], q[5];
rz(2.566865003044965) q[4];
cx q[2], q[8];
cx q[1], q[7];
rz(0.12525443480099505) q[0];
rz(6.013863160833178) q[3];
cx q[9], q[4];
cx q[6], q[5];
cx q[7], q[2];
rz(2.455298252105371) q[9];
rz(0.5449436566008138) q[0];
rz(3.8187691950594362) q[6];
cx q[3], q[5];
cx q[8], q[4];
rz(5.940957820407158) q[1];
rz(5.08230346753632) q[4];
cx q[8], q[6];
rz(2.8967059035075797) q[7];
cx q[9], q[0];
cx q[1], q[2];
cx q[3], q[5];
rz(5.626308307426205) q[7];
cx q[9], q[0];
rz(2.495312386432567) q[6];
rz(6.185459682270624) q[5];
rz(5.163091502090532) q[3];
cx q[2], q[1];
rz(1.8009197981855063) q[8];
rz(1.4015941192734387) q[4];
rz(5.104575029141691) q[8];
rz(2.060613139264746) q[3];
rz(1.9007134761280082) q[2];
rz(5.030951936131066) q[5];
rz(1.415114561718468) q[9];
rz(4.827667255898451) q[4];
rz(0.6591504580517513) q[6];
rz(3.7233962745539393) q[7];
rz(1.7180374098186562) q[0];
rz(5.242119337953764) q[1];
rz(5.961587343162028) q[1];
rz(4.669471035977542) q[9];
cx q[5], q[8];
cx q[7], q[4];
rz(2.6039552079111488) q[0];
rz(1.7598473431720116) q[6];
rz(0.12127162410228978) q[2];
rz(6.0430056151817455) q[3];
cx q[3], q[6];
rz(3.3155443904428124) q[9];
rz(1.0181324085327152) q[1];
rz(0.8705934972149311) q[8];
rz(5.556269604322559) q[5];
rz(4.258006685597661) q[7];
rz(5.813865041737672) q[2];
rz(5.77281603015101) q[0];
rz(0.6240975722770771) q[4];
cx q[2], q[9];
rz(2.401864371255027) q[4];
rz(4.002408559229569) q[3];
cx q[6], q[1];
rz(3.397474466615378) q[7];
rz(1.0199719119103476) q[8];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
cx q[5], q[0];
rz(0.5234810935510381) q[4];
rz(3.149610737793675) q[8];
rz(2.5847969551111047) q[2];
rz(2.008707038923432) q[0];
rz(3.7298295595498403) q[3];
rz(4.769573243514399) q[5];
rz(2.5091162679835515) q[6];
rz(6.158603216910978) q[7];
rz(0.18930652401631945) q[9];
rz(4.876108723191043) q[1];
rz(2.566237381808146) q[7];
rz(2.9262248863470584) q[6];
cx q[9], q[4];
cx q[1], q[2];
rz(4.265076192121759) q[0];
rz(5.753492525786306) q[3];
rz(1.5265431372559572) q[5];
rz(0.33072095642043414) q[8];
rz(0.6279704350733344) q[8];
rz(1.060002155193003) q[0];
rz(1.4534927729847265) q[6];
rz(3.3147211183596923) q[4];
rz(1.7655215941402833) q[5];
rz(4.718722097437966) q[3];
rz(1.3384797328100917) q[2];
rz(4.605486330792542) q[7];
rz(4.655711316699632) q[1];
rz(2.5850990811289107) q[9];
rz(0.6176395914310167) q[2];
rz(0.8284135096296809) q[0];
cx q[6], q[7];
rz(3.2040810853300394) q[8];
cx q[5], q[9];
rz(4.6447808352069275) q[4];
cx q[1], q[3];
cx q[8], q[2];
rz(5.8161677367394775) q[7];
rz(3.882173466931125) q[9];
rz(4.600563847636845) q[6];
rz(5.671023940770848) q[3];
rz(4.966967244387184) q[4];
cx q[0], q[5];
rz(0.09770566019915071) q[1];
rz(5.931130946532254) q[9];
rz(3.5269938833073478) q[1];
rz(0.805434445029265) q[5];
cx q[0], q[2];
cx q[4], q[8];
rz(2.1502928148107667) q[3];
rz(6.157343880985891) q[7];
rz(1.5017150995221458) q[6];
rz(5.0529911718286025) q[6];
rz(4.169339293856896) q[4];
rz(0.3319273795847448) q[8];
cx q[1], q[9];
cx q[0], q[3];
rz(4.542845642697389) q[5];
rz(4.662602326043881) q[7];
rz(4.279605372683122) q[2];
rz(0.20093395552246135) q[7];
cx q[3], q[4];
rz(2.989548182123102) q[9];
rz(3.874253812009906) q[5];
cx q[1], q[8];
rz(4.5755694725409946) q[2];
rz(2.2960553226058384) q[6];
rz(3.8555706918135666) q[0];
cx q[2], q[4];
rz(0.18054618737327044) q[0];
rz(3.723496282145998) q[7];
rz(2.9463774886169) q[5];
rz(5.019222674944784) q[1];
rz(2.92509172171637) q[9];
cx q[8], q[3];