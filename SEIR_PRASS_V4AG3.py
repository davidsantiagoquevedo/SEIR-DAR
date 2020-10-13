#!/usr/bin/python
# -*- coding:utf8 -*-

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import time
from datetime import datetime

"""
En este script se modifica la clase SEIR_PRASS ya trabajada eliminando el compartimento 
IMNP y agregando un compartimento IEMP para el estado Infeccioso leve de los contactos
expuestos rastreados. Esto permite definir de forma adecuada la transición al estado 
recuperado de los mismos

update: 
01/09/2020: Se simplificó el módulo plot_varios() asignando fechas automáticamente desde los
			datos, utilizando los módulos time y datetime
31/08/2020: Se agrega función set_beta(t) para definir beta desde afuera
25/08/2020: Se agregan compartimentos de seguimiento para sospechosos primarios
			contactos sospechosos, infecciosos primarios y contactos sospechosos
			y expuestos de infecciosos primarios
"""

########################################################################
############### ########## SEIR_PRASS CLASS ########## #################
########################################################################
class SEIR_PRASS():
	def __init__(self):
		self.dummy=True

	def set_trans(self,beta,beta_0,beta_1,beta_H,r,Movilidad=0):
		self.beta_0=beta_0
		self.beta_1=beta_1
		self.beta_H=beta_H
		self.r=r
		self.Movilidad=Movilidad
		self.beta=beta

	def var_t_estadia(self, omega, gamma_M, sigma_C, sigma_CA, sigma_CP,
					  gamma_HR, nu, gamma_R, sigma_HD, sigma_UD):
		self.omega=omega		#T. prom. latencia
		self.gamma_M=gamma_M	#T. prom de recuperación para IM
		self.sigma_C=sigma_C	#T. antes de aislamiento IC
		self.sigma_CA=sigma_CA	#T. en aislamiento ICA
		self.sigma_CP=sigma_CP	#T. de PRASSEADOS antes de Hospitalización
		self.gamma_HR=gamma_HR	#T. prom HR->R
		self.nu=nu				#T. prom HU->IR
		self.gamma_R=gamma_R	#T. prom IR->R
		self.sigma_HD=sigma_HD	#T. prom HD->D
		self.sigma_UD=sigma_UD	#T. prom UD->D

	def var_H_UCI(self, delta_M, delta_HR, delta_HD, delta_UR):
		self.delta_M=delta_M
		self.delta_HR=delta_HR
		self.delta_HD=delta_HD
		self.delta_UR=delta_UR
		self.delta_UD=1.-delta_HR-delta_HD-delta_UR

	def var_testeo(self, xi_PCR=0, xi_AG=0, T_PCR=1., T_AG=1., psi_T=1., psi_T0=False):
		self.xi_PCR=xi_PCR
		self.xi_AG=xi_AG
		self.T_PCR=T_PCR
		self.T_AG=T_AG
		self.psi_T=psi_T
		if psi_T0==False:
			self.psi_T0=psi_T
		else:
			self.psi_T0=psi_T0

	def var_PRASS(self, alpha, t0_PRASS=198, n=2, nr=0, rho=0, q=0): 
		self.alpha=alpha
		self.n=float(n)
		self.nr=float(nr)
		self.rho=rho
		self.q=q
		self.t0_PRASS=t0_PRASS

	def set_f_PRASS(self, f_theta,f_phi,f_phip):
		self.theta_t=f_theta
		self.phi_t=f_phi
		self.phip_t=f_phip

	def var_ini(self, N0, E0=0, EP0=0, IM0=0, IMP0=0, IEMP0=0, 
				IC0=0, ICP0=0, ICA0=0, IHR0=0, IUR0=0, IHD0=0, IUD0=0, 
				IR0=0, R0=0, D0=0,
				QIMAG10=0, QEP0=0, QEIM0=0, QEIM10=0, QEIC0=0, QS0=0, QSDC0=0):

		self.N0=N0
		self.E0=E0
		self.EP0=EP0

		self.IM0=IM0
		self.IMP0=IMP0
		self.IEMP0=IEMP0

		self.IC0=IC0
		self.ICP0=ICP0
		self.ICA0=ICA0

		self.IHR0=IHR0
		self.IUR0=IUR0
		self.IHD0=IHD0
		self.IUD0=IUD0
		self.IR0=IR0
		self.R0=R0
		self.D0=D0

		self.QIMAG10=QIMAG10
		self.QEP0=QEP0
		self.QEIM0=QEIM0
		self.QEIM10=QEIM10
		self.QEIC0=QEIC0
		self.QS0=QS0
		self.QSDC0=QSDC0

		self.S0= (self.N0 - self.E0 - self.EP0 - self.IM0 - self.IMP0  - self.IEMP0 
				  - self.IC0 - self.ICP0 - self.ICA0 - self.IHR0 - self.IUR0 - self.IHD0 
				  - self.IUD0 - self.IR0 - self.R0 - self.D0 - self.IM0
				  - self. QIMAG10 - self.QEP0 - self. QEIM0 - self.QEIM10 - self. QEIC0 - self.QS0 - self.QSDC0)

	def ODES(self,y,t):
		S, E, EP, IM, IMP, IEMP, IC, ICP, ICA, IHR, IUR, IHD, IUD, IR, R, D, N, QIMAG1, QEP, QEIM, QEIM1, QEIC, QS, QSDC = y
		theta=self.theta_t(t,IM)
		beta=self.beta(t,self.Movilidad)
		b=beta/self.n
		
		dSdt = (-S/float(N)*(beta*(IM+IC+(1.-self.r)*ICA+(1.-self.q)*(IEMP+IMP+ICP))+self.beta_H*(IHR+IUR+IHD+IUD+IR))
				- self.n*(1.-b)*self.phip_t(t)*theta*IM*S/float(N) +1./(1./self.omega + self.T_PCR)*QS 
				-(self.psi_T-1.)*(self.nr+1.)*self.alpha*theta*IM*S/float(N) + 1./self.T_AG*QSDC)

		dEdt =(beta*(1.-theta)*IM*S/float(N) + beta*(1.-self.phi_t(t))*theta*IM*S/float(N) 
			 + beta*S/float(N)*(IC+(1.-self.r)*ICA + (1-self.q)*(IEMP+ICP+IMP))+ self.beta_H*(IHR+IUR+IHD+IUD+IR)- self.omega*E)
		dEPdt = beta*self.phi_t(t)*theta*IM*S/float(N) - self.omega*EP

		dIMdt = self.delta_M*self.omega*E - theta*self.alpha*IM - (1.-theta)*self.gamma_M*IM
		dIMPdt = theta*self.alpha*IM - 1./(1./self.gamma_M-1./self.alpha)*IMP
		dIEMPdt = self.delta_M*self.omega*EP  - self.gamma_M*IEMP

		dICdt = (1.-self.delta_M)*self.omega*E - self.sigma_C*IC
		dICAdt = self.sigma_C*IC - self.sigma_CA*ICA
		dICPdt = (1.-self.delta_M)*self.omega*EP - self.sigma_CP*ICP

		dIHRdt = self.delta_HR*(self.sigma_CA*ICA+self.sigma_CP*ICP) - self.gamma_HR*IHR
		dIURdt = self.delta_UR*(self.sigma_CA*ICA+self.sigma_CP*ICP) - self.nu*IUR
		dIHDdt = self.delta_HD*(self.sigma_CA*ICA+self.sigma_CP*ICP) - self.sigma_HD*IHD
		dIUDdt = self.delta_UD*(self.sigma_CA*ICA+self.sigma_CP*ICP) - self.sigma_UD*IUD

		dIRdt = self.nu*IUR - self.gamma_R*IR
		dRdt = self.gamma_HR*IHR + self.gamma_R*IR + 1./(1./self.gamma_M-1./self.alpha)*IMP + self.gamma_M*IEMP + (1.-theta)*self.gamma_M*IM

		dDdt = self.sigma_HD*IHD + self.sigma_UD*IUD
		dNdt = -self.sigma_HD*IHD - self.sigma_UD*IUD

		# Cuarentena índices infecciosos
		dQIMAG1dt = self.xi_AG*theta*self.alpha*IM - self.rho*QIMAG1

		# Cuarentena contactos Expuestos
		dQEPdt = beta*self.phi_t(t)*theta*IM*S/float(N) - self.omega*QEP
		dQEIMdt = self.delta_M*self.omega*QEP - 1./float(self.T_PCR)*QEIM
		dQEIM1dt = self.xi_PCR*1./float(self.T_PCR)*QEIM - 1./((1./self.rho)-(1./self.omega)-self.T_PCR)*QEIM1
		dQEICdt = (1.-self.delta_M)*self.omega*QEP - self.sigma_CP*QEIC

		# Cuarentena contactos no expuestos + índices no infecciosos + contactos índices no infecciosos
		dQSdt = self.n*(1.-b)*self.phip_t(t)*theta*IM*S/float(N) - 1./(1./self.omega + self.T_PCR)*QS 
		dQSDCdt= (self.psi_T-1.)*(self.nr+1.)*self.alpha*theta*IM*S/float(N) - 1./self.T_AG*QSDC

		return [dSdt, dEdt, dEPdt, dIMdt, dIMPdt, dIEMPdt, dICdt, dICPdt, dICAdt, 
				dIHRdt, dIURdt, dIHDdt, dIUDdt, dIRdt, dRdt, dDdt, dNdt,
				dQIMAG1dt ,dQEPdt, dQEIMdt, dQEIM1dt, dQEICdt, dQSdt, dQSDCdt]

	def solve(self,t0,tf,dt):
		self.t0=t0
		self.tf=tf
		self.dt_=1/dt
		y0= [self.S0, self.E0, self.EP0, self.IM0, self.IMP0, self.IEMP0, 
			 self.IC0, self.ICP0, self.ICA0, self.IHR0, self.IUR0, 
			 self.IHD0, self.IUD0, self.IR0, self.R0, self.D0, self.N0,
			 self.QIMAG10, self.QEP0, self.QEIM0, self.QEIM10, self.QEIC0, self.QS0, self.QSDC0]

		t_vect= np.linspace(self.t0, self.tf, (self.tf-self.t0)*self.dt_+1)
		self.t_=t_vect
		solution= odeint(self.ODES,y0,t_vect)

		self.S_vect=solution.T[0]
		self.E_vect=solution.T[1]
		self.EP_vect=solution.T[2]

		self.IM_vect=solution.T[3]
		self.IMP_vect=solution.T[4]
		self.IEMP_vect=solution.T[5]

		self.IC_vect=solution.T[6]
		self.ICP_vect=solution.T[7]
		self.ICA_vect=solution.T[8]

		self.IHR_vect=solution.T[9]
		self.IUR_vect=solution.T[10]
		self.IHD_vect=solution.T[11]
		self.IUD_vect=solution.T[12]

		self.IR_vect=solution.T[13]
		self.R_vect=solution.T[14]
		self.D_vect=solution.T[15]
		self.N_vect=solution.T[16]

		self.QIMAG1_vect=solution.T[17]
		self.QEP_vect=solution.T[18]
		self.QEIM_vect=solution.T[19]
		self.QEIM1_vect=solution.T[20]
		self.QEIC_vect=solution.T[21]
		self.QS_vect=solution.T[22]
		self.QSDC_vect=solution.T[23]

	def Contar_Tasas_Prass(self):
		self.theta_vect=[]
		self.psi_vect=[]
		self.var_testeo(xi_PCR=self.xi_PCR, 
						xi_AG=self.xi_AG, 
						T_PCR=self.T_PCR,
						T_AG=self.T_AG,  
						psi_T=self.psi_T0, 
						psi_T0=self.psi_T0)
		for i in range(len(self.t_)):
			self.theta_vect.append(self.theta_t(self.t_[i],self.IM_vect[i]))
			self.psi_vect.append(self.psi_T)

	def Contar_PRASS(self):
		self.Contar_Tasas_Prass()
		self.PRASS_nuevos_indices=[]
		self.PRASS_aislamientos_diarios=[]
		self.PRASS_aislamientos_contactos=[]
		for i in range(len(self.t_)):
			theta=self.theta_vect[i]
			psi_T=self.psi_vect[i]
			beta=self.beta(self.t_[i],self.Movilidad)
			b=beta/self.n
			
			self.PRASS_nuevos_indices.append(theta*self.IM_vect[i]*psi_T)
			self.PRASS_aislamientos_diarios.append(self.QIMAG1_vect[i]+self.QEP_vect[i]+
												   self.QEIM_vect[i]+self.QEIM1_vect[i]+
												   self.QEIC_vect[i]+self.QS_vect[i]+
												   self.QSDC_vect[i])
			self.PRASS_aislamientos_contactos.append((self.n*(1-b)*self.phip_t(self.t_[i])*theta+(self.psi_T-1.)*(self.nr)*self.alpha*theta + beta*self.phi_t(self.t_[i])*theta)*self.S_vect[i]*self.IM_vect[i]/float(self.N_vect[i]))
			
	def Contar_Tests(self):
		self.PCR=[]
		self.AG=[]
		self.pos_PCR=[]
		self.pos_AG=[]
		self.Contar_Tasas_Prass()
		for i in range(len(self.t_)):
			theta=self.theta_vect[i]
			psi_T=self.psi_vect[i]
						
			if self.t_[i]<self.t0_PRASS:
				self.PCR.append(13000.)
				self.pos_PCR.append(13000./float(psi_T))
				self.AG.append(1)
				self.pos_AG.append(0)
			if self.t_[i]>=self.t0_PRASS:
				beta=self.beta(self.t_[i],self.Movilidad)
				b=beta/self.n
				
				self.PCR.append(self.omega*self.QEP_vect[i]+self.n*(1-b)*self.phip_t(self.t_[i])*theta*self.S_vect[i]*self.IM_vect[i]/float(self.N_vect[i]))
				self.pos_PCR.append(self.omega*self.QEP_vect[i]*self.xi_PCR)
				self.AG.append(self.sigma_CA*self.ICA_vect[i]*psi_T+theta*self.alpha*self.IM_vect[i]*psi_T)
				self.pos_AG.append((self.sigma_CA*self.ICA_vect[i]+theta*self.alpha*self.IM_vect[i])*self.xi_AG)
		self.R_pos_PCR=np.array(self.pos_PCR)/np.array(self.PCR)
		self.R_pos_AG=np.array(self.pos_AG)/np.array(self.AG)

	def Contar_UCI(self):
		self.Requieren_UCI=np.array(self.IUR_vect)+np.array(self.IUD_vect)

	def Calcular_Proporciones(self):
		self.contconf_conf=[]
		self.contais_ais=[]
		"""
		for i in range(len(self.t_)):
			if self.t_[i]<self.t0_PRASS:
				self.contconf_conf.append(0)
				self.contais_ais.append(0)
			else:
				self.contconf_conf.append(self.omega*self.EP_vect[i]/(self.PRASS_incidencia_IMP[i]+self.omega*self.EP_vect[i]))
				self.contais_ais.append((self.FEP_vect[i]+self.FEIM_vect[i]+self.FEIM1_vect[1]+
										 self.FCSIM_vect[i]+self.FCSS_vect[i])
										 /(self.PRASS_aislamientos_diarios[i]))
		"""
