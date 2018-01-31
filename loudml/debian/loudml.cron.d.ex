#
# Regular cron jobs for the loudml package
#
0 4	* * *	root	[ -x /usr/bin/loudml_maintenance ] && /usr/bin/loudml_maintenance
