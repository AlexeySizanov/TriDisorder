## Первый порядок через спины

$$
\newcommand{\ch}{{\cal H}}
\newcommand{\a}{\alpha}
\newcommand{\b}{\beta}
\newcommand{\ve}{\varepsilon}
\newcommand{\pd}{\partial}
\newcommand{\ab}{{\alpha\beta}}
\newcommand{\vS}{\vec{S}}
\newcommand{\s}{\vec{S}}
\newcommand{\la}{\lambda}
\newcommand{\vmu}{\vec{\mu}}
\newcommand{\vve}{\vec{\varepsilon}}
\newcommand{\beq}{\begin{eqnarray}}
\newcommand{\eeq}{\end{eqnarray}}
\newcommand{\vphi}{\varphi}
\nonumber
$$

$$
\ch'=\frac12 J_{ab}\s_a\s_b + \vmu_c \la_c \s_c
$$

Условие равновесия:
$$
\begin{eqnarray}
\pd_{bz} \ch' &=& \left (J_{ab}\s_a + \vmu_b \la_b \right) \pd_z \s_b = 0, \\
\pd_{\la_c} \ch' &=& \vmu_c \s_c = 0
\end{eqnarray}
$$
Скрутка:
$$
\begin{eqnarray}\vmu_c &\rightarrow& \vmu_c +  \vve_c, \\\theta_a &\rightarrow& \theta_a + x_a, \\\vphi_a &\rightarrow& \vphi_a + y_a\end{eqnarray}
$$
Раскладываем всё в ряд по $\ve$:
$$
\left[
    J_{ab} \left(  
        \s_a + \pd_{z'} \s_a y_{az'}
    \right) +
   	\left( \vmu_b + \vve_b	\right)
   	\left( \la_b + \delta \la_b \right)
\right]
\left( 
\pd_z \s_b + \pd_{zz'} \s_b y_{bz'}
\right) = 0_{bz} \\

(\vmu_c + \vve_c) \left( \s_c + \pd_z \s_c y_{zc}\right) = 0_c
$$
Первый порядок:
$$
\left[
		\left( J_{cb} \s_c  + \vmu_b \la_b \right)_b  \pd_{zz'} \s_b \, \delta_{ab} + 
		J_{ab} \pd_{z'} \s_a \pd_z \s_b
\right]_{bzaz'} y_{az'}
+ 
\left( \vmu_b \pd_z \s_b \right)_{bz} \delta \la_b 
=
- \left( \vve_b \la_b \pd_z \s_b \right)_{bz} \\

\vmu_c \pd_z \s_c y_{cz} + \vve_c \s_c = 0_c
$$
