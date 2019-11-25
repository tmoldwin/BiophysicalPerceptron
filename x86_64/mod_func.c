#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _CaDynamics_E2_reg(void);
extern void _Ca_HVA_reg(void);
extern void _Ca_LVAst_reg(void);
extern void _GluSynapse_reg(void);
extern void _Ih_reg(void);
extern void _Im_reg(void);
extern void _K_Pst_reg(void);
extern void _K_Tst_reg(void);
extern void _NaTa_t_reg(void);
extern void _NaTs2_t_reg(void);
extern void _Nap_Et2_reg(void);
extern void _ProbAMPANMDA_EMS_reg(void);
extern void _ProbAMPANMDA_EMS_Current_reg(void);
extern void _ProbGABAAB_EMS_reg(void);
extern void _ProbGABAAB_EMS_Current_reg(void);
extern void _SK_E2_reg(void);
extern void _SKv3_1_reg(void);
extern void _Z_g_reg(void);
extern void _epsp_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," mod/CaDynamics_E2.mod");
    fprintf(stderr," mod/Ca_HVA.mod");
    fprintf(stderr," mod/Ca_LVAst.mod");
    fprintf(stderr," mod/GluSynapse.mod");
    fprintf(stderr," mod/Ih.mod");
    fprintf(stderr," mod/Im.mod");
    fprintf(stderr," mod/K_Pst.mod");
    fprintf(stderr," mod/K_Tst.mod");
    fprintf(stderr," mod/NaTa_t.mod");
    fprintf(stderr," mod/NaTs2_t.mod");
    fprintf(stderr," mod/Nap_Et2.mod");
    fprintf(stderr," mod/ProbAMPANMDA_EMS.mod");
    fprintf(stderr," mod/ProbAMPANMDA_EMS_Current.mod");
    fprintf(stderr," mod/ProbGABAAB_EMS.mod");
    fprintf(stderr," mod/ProbGABAAB_EMS_Current.mod");
    fprintf(stderr," mod/SK_E2.mod");
    fprintf(stderr," mod/SKv3_1.mod");
    fprintf(stderr," mod/Z_g.mod");
    fprintf(stderr," mod/epsp.mod");
    fprintf(stderr, "\n");
  }
  _CaDynamics_E2_reg();
  _Ca_HVA_reg();
  _Ca_LVAst_reg();
  _GluSynapse_reg();
  _Ih_reg();
  _Im_reg();
  _K_Pst_reg();
  _K_Tst_reg();
  _NaTa_t_reg();
  _NaTs2_t_reg();
  _Nap_Et2_reg();
  _ProbAMPANMDA_EMS_reg();
  _ProbAMPANMDA_EMS_Current_reg();
  _ProbGABAAB_EMS_reg();
  _ProbGABAAB_EMS_Current_reg();
  _SK_E2_reg();
  _SKv3_1_reg();
  _Z_g_reg();
  _epsp_reg();
}
