## Оптимизация с деформацией.

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
\nonumber
$$

Исходный гамильтониан
$$
{\cal H} = \frac12 J_{ab} \vec{S}_a \vec{S}_b
$$
где индексы $a,\,b$  нумеруют все спины. Чтобы ввести ограничения на противоположных плоскостях добавим множители Лагранжа:
$$
{\cal H'} = {\cal H} + \vec{\mu}\lambda_{c} \vec{S}_c + \vec{\nu} \lambda_d \vec{S}_d
$$
где $\vec{\mu}, \, \vec{\nu}$ – нормали к спиновым плоскостям на противоположных пространственных плоскостях, спины на которых нумеруют индексы $c,\, d$. Функции $\cal H,\, H'$ можно представит в виде
$$
\begin{eqnarray}
{\cal H} &=& \frac12 A_\ab \, w_\a \, w_\b,\\
{\cal H'} &=& \frac12 A'_{\alpha\beta} \, w_\alpha \, w_\beta,\\
w &=& (S^x_a,\, S^y_a,\, S^z_a,\, \lambda_c,\, \lambda_d)
\end{eqnarray}
$$

Мы будем скручивать одну границу:

$$
\begin{eqnarray}
\vec{\nu} &\rightarrow& \vec{\nu} + \vec{\varepsilon}, \quad \vec{\varepsilon} \perp \vec{\nu} \\
A &\rightarrow& A,\\
A' &\rightarrow& A' + \varepsilon_\eta A^\eta
\end{eqnarray}
$$

Условие равновесия вблизи старого минимума
$$
A'_\ab w_\a \pd_k w_\b = 0 \label{eq}
$$
Считая
$$
y_i = y^\eta_i \ve_\eta + \frac12 y^{\eta\rho} \ve_\eta \ve_\rho,
$$
раскладываем $\eqref{eq}$ по степеням $\ve$:
$$
\left( A'_\ab + \ve_\eta A^\eta_\ab \right)_\ab * \\
\left[ 
	w_\a + 
	\left(\pd_i w_\a y^\eta_i \right) \ve_\eta +
	\frac12 \left( 
		\pd_i w_\a y^{\eta\rho}_i + \pd_{ij}w_a y^\eta_i y^\rho_j
	\right) \ve_\eta \ve_\rho
\right]_\a *\\

\left[ 
	\pd_k w_\b + 
	\left(\pd_{ki} w_\b y^\eta_i \right) \ve_\eta +
	\frac12 \left( 
		\pd_{ki} w_\b y^{\eta\rho}_i + \pd_{kij}w_\b y^\eta_i y^\rho_j
	\right) \ve_\eta \ve_\rho
\right]_{\b k} = 0
$$
Получаем уравнения, для $\ve^1$:
$$
\left[
A'_\ab \left(
	w_\a \pd_{ki} w_\b + \pd_k w_\b \pd_i w_\a
\right)
\right]_{ki} y^\eta_i = 

- \left( A^\eta_\ab w_\a\pd_k w_\b \right)^\eta_k

\label{order1}
$$
и для $\ve^2$:
$$
\frac12 A'_\ab \left( 
	w_\a \pd_{ki} w_\b + \pd_k w_\b \pd_i w_a
\right) y^{\eta\rho}_i = \\

- \frac12 A'_\ab \left(
2\pd_i w_\a \pd_{kj} w_\b +
	w_\a \pd_{kij} w_\b + 
	\pd_k w_\b \pd_{ij} w_\a
\right) y^\eta_i y^\rho_j\\

- A^\eta_\ab \left( 
	w_\a \pd_{ki} w_\b + \pd_k w_\b \pd_i w_\a
\right) y^\eta_i

\label{order2}
$$
Уравнения $\eqref{order1},\; \eqref{order2}$  решаются последовательно методом наименьших квадратов (так как они недоопределены и обратной матрицы не сущетсвует).



Производные исходного гамильтониана:
$$
\begin{eqnarray}
\frac{\pd{\cal H}}{\pd \ve_\eta} &=& 
	\pd_i {\cal H} y^\eta_i = 
	A_\ab w_\a \pd_i w_b y^\eta_i \\
\frac{\pd^2 {\cal H}}{\pd \ve_\eta \pd \ve_\rho} &=&
	A_\ab \left( 
		\pd_i w_a \pd_j w_b  + w_\a \pd_{ij} w_\b
	\right) y^\eta_i y^\rho_j + 
	
  \pd_i {\cal H} y^{\eta\rho}_i \label{diff2}
\end{eqnarray}
$$
Так как мы руками фиксируем спиновые плоскости на границах, то $\pd_i {\cal H'} = 0$, но $\pd_i {\cal H} \ne 0$  и мы не можем отбросить последний член в $\eqref{diff2}$, что вынуждает нас вычислять $y^{\eta\rho}$, пользуясь уравнением $\eqref{order2}$.

#### Отдельно по компонентам

$$
{\cal H'} = \frac12 J_{ab} \s_a \s_b + \la_d \vmu_d \s_d
$$

Условие равновесия:
$$
\begin{eqnarray}
\left[ J_{ac} \s_a  + \la_c \vmu_c \right] (\pd_z \s)_c &=& 0_{cz}
\\
\vmu_d \s_d &=& 0_d
\end{eqnarray}
$$
Деформируем граничные условия $\vmu_d \rightarrow \vmu_d + \ve \vec{e}_d$,  и разложимся до степени $\ve^2$:
$$
\vmu_d \rightarrow \vmu_d + \ve \vec{e}_d, \\
y_{cz} = y'_{cz} \ve + \frac12 y''_{cz} \ve^2, \\
\la_d \rightarrow \la_d + \la'_d \ve + \frac12 \la''_d \ve^2; \\
$$

$$
\begin{eqnarray}
\left[
	J_{ac}
	\left(
		\s_a + (\pd_v \s)_a y_{av} + \tfrac12 (\pd_{vu} \s)_a y_{av} y_{au}	
	\right)
	+
	\left(
		\la_c + \la'_c \ve + \tfrac12 \la''_c \ve^2
	\right)

	\left( \vmu_c + \ve \vec{e}_c \right) 
\right] * \quad \quad&&\\

\left[
	(\pd_z \s)_c + 
	(\pd_{zv} \s)_c y_{cv} +
	\tfrac12 (\pd_{zvu} \s)_c y_{cv} y_{cu}
\right] &=& 0_{cz} \\

\left( \vmu_d + \ve \vec{e}_d \right)
\left[\s_d + (\pd_v \s)_d y_{dv} + \tfrac12 (\pd_{vu} \s)_d y_{dv} y_{du}\right] &=& 0_d
\end{eqnarray}
$$
Порядок $\ve^1$:
$$
\begin{eqnarray}

\left[
	J_{ac} (\pd_v \s)_a y'_{av} + \la'_c \vmu_c + \la_c \vec{e}_c
\right] (\pd_z \s)_c + 

\left(
	J_{ac} \s_a + \la_c \vmu_c
\right) (\pd_{zv} \s)_c y'_{cv} &=& 0_{cz} \\

\vec{e}_d \s_d + \vmu_d (\pd_v \s)_d y'_{dv} &=& 0_d

\end{eqnarray}
$$

выделяем главное:
$$
\begin{eqnarray}
J_{ac} (\pd_v \s)_a y'_{av} + (J_{ac} \s_a + \la_c \vmu_c)(\pd_{zv} \s)_c y'_{cv} +
(\pd_z \s)_c \la'_c \vmu_c &=& 
- \la_c \vec{e}_c (\pd_z \s)_c \\

\vmu_d (\pd_v \s)_d y'_{dv} &=& - \vec{e}_d \s_d
\end{eqnarray}
$$

