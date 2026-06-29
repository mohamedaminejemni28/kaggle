function get_boxplot(x1,x2,varname,group,save, folder_name)
% plotting boxplot
x = [x1; x2];
g = [repmat({'Young'},length(x1),1); repmat({'Older'},length(x2),1)];
boxplot(x, g)
% ylabel('Unit')
title(varname, 'Interpreter', 'none')
% legend(group)
original_directory = pwd;
if save
    if exist(folder_name, 'dir') == 7
        cd(folder_name);
    end
    %     saveas(fig,'MySimulinkDiagram.bmp');
    print(gcf,strcat(varname,'_stats','.png'),'-dpng','-r50');
    pause(0.1)
    cd(original_directory)
end