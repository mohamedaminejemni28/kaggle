function get_histogram_groups(x1,x2,varname,group,save, folder_name)
h1 = x1';
h2 = x2';
histogram(h1,'facealpha',.7,'edgecolor','none');
hold on;
histogram(h2,'facealpha',.3,'edgecolor','none');
xlabel('Value')
ylabel('Count')
legend(group)
title(varname, 'Interpreter', 'none')
original_directory = pwd;
if save
    if exist(folder_name, 'dir') == 7
        cd(folder_name);
    end
    %     saveas(fig,'MySimulinkDiagram.bmp');
    print(gcf,strcat(varname,'_hist','.png'),'-dpng','-r50');
    cd(original_directory)
end
hold off
end